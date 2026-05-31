#!/usr/bin/env python3
"""ROS 2 AI lane keeper using JetRacer/HSHL ResNet road-following weights.

This node is intentionally isolated from the classical IPM/histogram stack. It
subscribes directly to the camera, runs a ResNet regression model, and publishes
Twist commands only while /system/current_state == LANE_KEEP. During obstacle
avoidance states, DWA remains the sole /cmd_vel writer.
"""
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

import cv2

from qcar2_autonomy.ai_lane_keeping.model import load_resnet_weights
from qcar2_autonomy.ai_lane_keeping.preprocess import preprocess_bgr_for_resnet


LANE_KEEP = 'LANE_KEEP'


class AiLaneKeeperNode(Node):
    """Camera-to-steering neural lane keeper with ROS safety gating."""

    def __init__(self):
        super().__init__('ai_lane_keeper_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('state_topic', '/system/current_state')
        self.declare_parameter('steering_topic', '/ai_lane_keeper/steering')
        self.declare_parameter('debug_image_topic', '/ai_lane_keeper/debug_image')
        self.declare_parameter('debug_mode', True)
        self.declare_parameter('model_path', '')
        self.declare_parameter('architecture', 'resnet18')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('output_mode', 'xy')  # xy or steering
        self.declare_parameter('input_width', 224)
        self.declare_parameter('input_height', 224)
        self.declare_parameter('crop_top_ratio', 0.0)
        self.declare_parameter('base_speed', 0.22)
        self.declare_parameter('min_speed', 0.10)
        self.declare_parameter('max_angular', 1.2)
        self.declare_parameter('steering_gain', 1.0)
        self.declare_parameter('steering_bias', 0.0)
        self.declare_parameter('steering_sign', -1.0)
        # Lateral offset added to the model's predicted x BEFORE the atan2
        # steering conversion. Use this when the network was trained to track a
        # LINE (e.g. the dashed middle line) and you want the car to drive
        # parallel to that line at a fixed sideways offset (i.e. inside a lane,
        # not on the line). Units are normalized image x in [-1, +1]: positive
        # = steer right of the predicted point, negative = steer left.
        self.declare_parameter('lane_offset', 0.0)
        self.declare_parameter('turn_slowdown_gain', 0.45)
        self.declare_parameter('image_timeout_sec', 0.5)
        self.declare_parameter('control_rate_hz', 20.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.architecture = self.get_parameter('architecture').value
        self.device_name = self.get_parameter('device').value
        self.output_mode = self.get_parameter('output_mode').value.strip().lower()
        self.input_width = int(self.get_parameter('input_width').value)
        self.input_height = int(self.get_parameter('input_height').value)
        self.crop_top_ratio = float(self.get_parameter('crop_top_ratio').value)
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_angular = abs(float(self.get_parameter('max_angular').value))
        self.steering_gain = float(self.get_parameter('steering_gain').value)
        self.steering_bias = float(self.get_parameter('steering_bias').value)
        self.steering_sign = float(self.get_parameter('steering_sign').value)
        self.lane_offset = float(self.get_parameter('lane_offset').value)
        self.turn_slowdown_gain = max(0.0, float(self.get_parameter('turn_slowdown_gain').value))
        self.image_timeout_sec = float(self.get_parameter('image_timeout_sec').value)

        if not self.model_path:
            raise ValueError('model_path must point to a ResNet .pth file')
        if self.output_mode not in ('xy', 'steering'):
            raise ValueError("output_mode must be 'xy' or 'steering'")
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError('input_width and input_height must be positive')
        if self.max_angular <= 0.0:
            raise ValueError('max_angular must be positive')

        output_dim = 2 if self.output_mode == 'xy' else 1
        self.model, self.device = load_resnet_weights(
            self.model_path,
            architecture=self.architecture,
            output_dim=output_dim,
            device=self.device_name,
        )

        # Import torch after load_resnet_weights so this module stays importable
        # on development machines without PyTorch.
        import torch
        self.torch = torch

        self.bridge = CvBridge()
        self.current_state = LANE_KEEP
        self.latest_bgr = None
        self.latest_image_time = None
        self.last_warning_time = None

        self.debug_mode = bool(self.get_parameter('debug_mode').value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.steering_pub = self.create_publisher(
            Float32,
            self.get_parameter('steering_topic').value,
            10,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            10,
        )
        self.create_subscription(Image, self.image_topic, self._on_image, 10)
        self.create_subscription(String, self.state_topic, self._on_state, 10)

        rate = max(1.0, float(self.get_parameter('control_rate_hz').value))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'AI lane keeper ready: model={self.model_path}, arch={self.architecture}, '
            f'device={self.device}, image={self.image_topic}, cmd={self.cmd_vel_topic}'
        )

    def _on_state(self, msg: String):
        new_state = msg.data.strip().upper() or LANE_KEEP
        if new_state != self.current_state:
            self.get_logger().info(f'state transition {self.current_state} -> {new_state}')
            if new_state != LANE_KEEP:
                self._publish_stop()
        self.current_state = new_state

    def _on_image(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image_time = self.get_clock().now()
        except CvBridgeError as exc:
            self.get_logger().warn(
                f'cv_bridge conversion failed: {exc}',
                throttle_duration_sec=2.0,
            )

    def _tick(self):
        # Run inference whenever we have a fresh frame, regardless of state.
        # This keeps the debug image alive during LANE_CHANGE so the user can
        # see what the AI "would have done" while DWA is driving. Twist is
        # only published when state == LANE_KEEP (below).
        outputs = None
        omega = 0.0
        infer_ok = False
        if self.latest_bgr is not None and not self._image_is_stale():
            try:
                outputs = self._run_inference(self.latest_bgr)
                omega = self._outputs_to_angular_z(outputs)
                infer_ok = True
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                self.get_logger().warn(
                    f'AI inference failed: {exc}',
                    throttle_duration_sec=2.0,
                )

        # Actuator publish path: only when this node owns /cmd_vel.
        if self.current_state == LANE_KEEP:
            if not infer_ok:
                self._publish_stop()
            else:
                speed = self._speed_for_turn(omega)
                out = Twist()
                out.linear.x = float(speed)
                out.angular.z = float(omega)
                self.cmd_pub.publish(out)

                steering_msg = Float32()
                steering_msg.data = float(omega)
                self.steering_pub.publish(steering_msg)

        # Debug image publish: always, when enabled and we have a frame.
        if self.debug_mode and self.latest_bgr is not None:
            self._publish_debug_image(self.latest_bgr, outputs, omega, infer_ok)

    def _image_is_stale(self):
        if self.latest_image_time is None:
            return True
        age = (self.get_clock().now() - self.latest_image_time).nanoseconds * 1e-9
        return age > self.image_timeout_sec

    def _run_inference(self, bgr):
        chw = preprocess_bgr_for_resnet(
            bgr,
            input_width=self.input_width,
            input_height=self.input_height,
            crop_top_ratio=self.crop_top_ratio,
        )
        tensor = self.torch.from_numpy(chw).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(output)):
            raise FloatingPointError(f'non-finite model output: {output}')
        return output

    def _outputs_to_angular_z(self, outputs):
        if self.output_mode == 'xy':
            if outputs.size < 2:
                raise ValueError(f'xy mode expected 2 outputs, got {outputs.size}')
            return self.steering_from_xy(
                x=float(outputs[0]),
                y=float(outputs[1]),
                gain=self.steering_gain,
                bias=self.steering_bias,
                sign=self.steering_sign,
                max_angular=self.max_angular,
                lane_offset=self.lane_offset,
            )

        if outputs.size < 1:
            raise ValueError('steering mode expected 1 output')
        omega = self.steering_sign * self.steering_gain * float(outputs[0]) + self.steering_bias
        return self._clamp(omega, -self.max_angular, self.max_angular)

    @staticmethod
    def steering_from_xy(
        x,
        y,
        gain=1.0,
        bias=0.0,
        sign=-1.0,
        max_angular=1.2,
        lane_offset=0.0,
    ):
        """Convert JetRacer direction-vector output into ROS angular.z.

        The trained network predicts a point/direction in image coordinates. The
        angle of that vector is:

            angle = atan2(x, y)

        Dividing by pi/2 maps a 90-degree camera error to +/-1. The default
        sign is negative because image x to the right should become negative
        ROS yaw rate (right turn). If your physical car turns the wrong way,
        launch with ``steering_sign:=1.0``.

        lane_offset is added to the model's predicted x before atan2. It is the
        first calibration knob for the "model tracks the lane marking, not the
        lane center" problem. With the default steering_sign=-1:

            lane_offset < 0  biases the vehicle left of the learned target.
            lane_offset > 0  biases the vehicle right of the learned target.

        Use small steps, usually 0.05 to 0.20.
        """
        if not math.isfinite(x) or not math.isfinite(y):
            raise FloatingPointError(f'non-finite xy output: x={x}, y={y}')
        x_target = x + float(lane_offset)
        angle = math.atan2(x_target, max(abs(y), 1.0e-6) * (1.0 if y >= 0.0 else -1.0))
        normalized = angle / (0.5 * math.pi)
        omega = sign * gain * normalized + bias
        return AiLaneKeeperNode._clamp(omega, -abs(max_angular), abs(max_angular))

    def _speed_for_turn(self, omega):
        turn_fraction = min(1.0, abs(omega) / self.max_angular)
        speed = self.base_speed * (1.0 - self.turn_slowdown_gain * turn_fraction)
        return self._clamp(speed, self.min_speed, self.base_speed)

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _publish_debug_image(self, bgr, outputs, omega, infer_ok):
        """Annotate the input frame with what the AI is seeing/predicting.

        Overlays:
          - red vertical line at image center
          - yellow circle at the predicted (x, y) point in image coordinates,
            mapped from the model's [-1, 1] output range
          - green line from the bottom-center to that point (the "look-at"
            direction the model is steering toward)
          - HUD text: model raw (x, y), final wz, current state
        """
        try:
            annotated = bgr.copy()
            h, w = annotated.shape[:2]

            # Reference markers
            cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)

            if infer_ok and outputs is not None and outputs.size >= 2:
                x_raw = float(outputs[0])
                y_raw = float(outputs[1])
                x_target = x_raw + self.lane_offset
                # Map model (x, y) in roughly [-1, 1] to image coordinates.
                # x: -1 = left edge, +1 = right edge.
                # y: -1 = bottom of image, +1 = top (model "look-at" is up the road).
                px_raw = int(max(0, min(w - 1, w // 2 + x_raw * (w // 2))))
                px = int(max(0, min(w - 1, w // 2 + x_target * (w // 2))))
                py = int(max(0, min(h - 1, h // 2 - y_raw * (h // 2))))
                cv2.line(annotated, (w // 2, h - 1), (px, py), (0, 255, 0), 2)
                cv2.circle(annotated, (px_raw, py), 7, (0, 180, 255), 1)
                cv2.circle(annotated, (px, py), 10, (0, 255, 255), 2)
                hud = (f'state={self.current_state}  raw(x,y)=({x_raw:+.2f},{y_raw:+.2f})  '
                       f'offset={self.lane_offset:+.2f}  wz={omega:+.2f}rad/s')
            else:
                hud = f'state={self.current_state}  (no inference)'

            cv2.putText(annotated, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.debug_image_pub.publish(msg)
        except (cv2.error, CvBridgeError) as exc:
            self.get_logger().warn(
                f'debug image publish failed: {exc}',
                throttle_duration_sec=2.0,
            )

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))


def main(args=None):
    rclpy.init(args=args)
    node = AiLaneKeeperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
