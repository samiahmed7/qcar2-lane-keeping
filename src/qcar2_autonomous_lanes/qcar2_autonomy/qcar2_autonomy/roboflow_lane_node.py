#!/usr/bin/env python3
"""Roboflow serverless lane segmentation for the QCar2 pipeline.

Calls a hosted YOLO-seg model on Roboflow (default: ``car-track-nkz9u/3``)
to segment the drivable lane area in the camera feed. Inference runs on a
background thread so the ~1.5 s cloud round-trip never blocks ROS callbacks.

Topics:
    /roboflow_lane/debug_image    sensor_msgs/Image   colored polygon overlay
    /roboflow_lane/predictions    std_msgs/String     JSON of raw predictions
    /planning/validated_target_x  std_msgs/Float32    lane-center pixel x
                                                      (consumed by PID follower)

The Roboflow model used here predicts the lane *interior* as a single mask
(rather than per-line instances). target_x is the x centroid of the mask near
the bottom of the image - the closest-to-the-car part of the lane.

Caveats:
    - Cloud latency is ~1.5 s on CPU networks. Driving will be coarse.
    - Roboflow free-tier has API call quotas. The infer_period_sec parameter
      caps how often we hit the endpoint.
    - Requires internet. No fallback.
"""
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from qcar2_autonomy.image_msg_utils import bgr_to_image_msg, image_msg_to_bgr
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


class RoboflowLaneNode(Node):
    """Threaded Roboflow inference + lane-center target publisher."""

    def __init__(self):
        super().__init__('roboflow_lane_node')

        # API
        self.declare_parameter('api_url', 'https://serverless.roboflow.com')
        self.declare_parameter('api_key', 'zevcoEtrBk9labeRvnWz')
        self.declare_parameter('model_id', 'car-track-nkz9u/3')
        # Throttle - don't blast the API faster than this. Cloud RTT alone is
        # ~1.5 s so anything tighter just queues + wastes quota.
        self.declare_parameter('infer_period_sec', 1.0)

        # Topics
        self.declare_parameter('image_topic', '/qcar2/front_camera/image')
        self.declare_parameter('debug_image_topic', '/roboflow_lane/debug_image')
        self.declare_parameter('predictions_topic', '/roboflow_lane/predictions')
        self.declare_parameter('target_topic', '/planning/validated_target_x')

        # Visualization / control tuning
        self.declare_parameter('publish_target', True)
        self.declare_parameter('lane_class_name', 'lane2')
        self.declare_parameter('mask_alpha', 0.45)
        # Bottom band of the image used to compute target_x. Smaller = closer
        # to the car = more reactive but noisier. Larger = smoother, slower to
        # respond to incoming curves.
        self.declare_parameter('target_band_y_start_ratio', 0.70)
        # Optional sideways nudge on the computed target (px in original frame).
        # Use this if the mask consistently biases the car off-center.
        self.declare_parameter('target_offset_px', 0.0)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.api_url = str(self.get_parameter('api_url').value)
        self.api_key = str(self.get_parameter('api_key').value)
        self.model_id = str(self.get_parameter('model_id').value)
        self.infer_period_sec = float(self.get_parameter('infer_period_sec').value)
        self.publish_target = bool(self.get_parameter('publish_target').value)
        self.lane_class_name = str(self.get_parameter('lane_class_name').value)
        self.mask_alpha = float(self.get_parameter('mask_alpha').value)
        self.target_band_y_start_ratio = float(
            self.get_parameter('target_band_y_start_ratio').value
        )
        self.target_offset_px = float(self.get_parameter('target_offset_px').value)

        # Import the SDK lazily so other nodes in the package don't get a hard
        # dependency on inference-sdk.
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError(
                'inference-sdk required. pip install inference-sdk'
            ) from exc

        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key,
        )
        self.get_logger().info(
            f'Roboflow client ready: model_id={self.model_id} url={self.api_url}'
        )

        # Shared state between the camera callback, worker thread, and timer.
        self._state_lock = threading.Lock()
        self.latest_bgr = None             # newest camera frame (BGR uint8)
        self.latest_overlay = None         # newest overlay to publish
        self.latest_target_x = None        # newest computed target_x
        self.latest_predictions_json = '[]'
        self._stop_event = threading.Event()
        self._last_infer_ms = 0

        # Publishers / subscribers
        self.image_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, 10
        )
        self.target_pub = self.create_publisher(
            Float32, self.get_parameter('target_topic').value, 10
        )
        self.predictions_pub = self.create_publisher(
            String, self.get_parameter('predictions_topic').value, 10
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image,
            10,
        )

        # The publish timer just emits cached overlay/target. The slow part
        # (cloud inference) runs in the worker thread below.
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / rate, self._publish_tick)

        self.worker = threading.Thread(target=self._infer_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            'Roboflow lane node up: '
            f'image={self.get_parameter("image_topic").value} -> '
            f'overlay={self.get_parameter("debug_image_topic").value}, '
            f'target={self.get_parameter("target_topic").value}, '
            f'infer_period={self.infer_period_sec:.2f}s'
        )

    def _on_image(self, msg: Image):
        try:
            bgr = image_msg_to_bgr(msg)
        except (cv2.error, ValueError) as exc:
            self.get_logger().warn(
                f'image convert failed: {exc}', throttle_duration_sec=2.0,
            )
            return
        with self._state_lock:
            self.latest_bgr = bgr

    def _infer_loop(self):
        """Worker loop: pull newest frame, run inference, cache overlay + target."""
        while not self._stop_event.is_set():
            with self._state_lock:
                frame = self.latest_bgr.copy() if self.latest_bgr is not None else None
            if frame is None:
                time.sleep(0.1)
                continue

            tmp_path = '/tmp/_roboflow_in.jpg'
            cv2.imwrite(tmp_path, frame)
            t0 = time.monotonic()
            try:
                result = self.client.infer(tmp_path, model_id=self.model_id)
            except Exception as exc:  # noqa: BLE001 - the SDK throws several types
                self.get_logger().warn(
                    f'Roboflow inference failed: {exc}',
                    throttle_duration_sec=2.0,
                )
                time.sleep(self.infer_period_sec)
                continue
            self._last_infer_ms = int(round((time.monotonic() - t0) * 1000))

            overlay, target_x = self._render_and_target(frame, result)
            preds_json = json.dumps(result.get('predictions', []))

            with self._state_lock:
                self.latest_overlay = overlay
                self.latest_target_x = target_x
                self.latest_predictions_json = preds_json

            # Sleep the remainder of the period.
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self.infer_period_sec - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _render_and_target(self, frame, result):
        """Draw the polygon mask and compute target_x from its bottom centroid."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        target_x = None
        target_y = None

        preds = result.get('predictions', []) or []
        # Pick the highest-confidence lane2 prediction. If your model class
        # name changes, update the lane_class_name parameter.
        lane_pred = None
        for p in preds:
            if p.get('class') == self.lane_class_name:
                if lane_pred is None or p.get('confidence', 0) > lane_pred.get('confidence', 0):
                    lane_pred = p

        if lane_pred is not None and 'points' in lane_pred:
            pts_xy = lane_pred['points']
            poly = np.array(
                [[int(round(pt['x'])), int(round(pt['y']))] for pt in pts_xy],
                dtype=np.int32,
            )

            # Fill mask for centroid math.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            # Bottom band centroid -> target_x.
            band_top = int(self.target_band_y_start_ratio * h)
            band = mask[band_top:, :]
            if band.any():
                ys, xs = np.where(band > 0)
                target_x = float(np.mean(xs)) + self.target_offset_px
                target_x = max(0.0, min(float(w - 1), target_x))
                target_y = band_top + int(np.mean(ys))

            # Color overlay.
            color_layer = overlay.copy()
            cv2.fillPoly(color_layer, [poly], (0, 255, 255))   # yellow fill
            overlay = cv2.addWeighted(color_layer, self.mask_alpha,
                                       overlay, 1.0 - self.mask_alpha, 0.0)
            cv2.polylines(overlay, [poly], True, (0, 200, 255), 2)
            cv2.putText(
                overlay,
                f'{self.lane_class_name} {lane_pred["confidence"]:.2f}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )

        # Camera center reference + target marker.
        cv2.line(overlay, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)
        if target_x is not None:
            cv2.line(overlay, (int(target_x), 0), (int(target_x), h),
                     (255, 0, 255), 2)
            if target_y is not None:
                cv2.circle(overlay, (int(target_x), int(target_y)), 8,
                           (255, 0, 255), -1)

        # HUD
        n = len(preds)
        tx_str = f'{target_x:.1f}' if target_x is not None else 'none'
        hud = (f'Roboflow {self.model_id}   infer: {self._last_infer_ms}ms   '
               f'preds: {n}   target_x={tx_str}')
        cv2.putText(overlay, hud, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(overlay, hud, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return overlay, target_x

    def _publish_tick(self):
        with self._state_lock:
            overlay = self.latest_overlay
            target_x = self.latest_target_x
            preds_json = self.latest_predictions_json

        if overlay is not None:
            try:
                self.image_pub.publish(bgr_to_image_msg(overlay))
            except (cv2.error, ValueError) as exc:
                self.get_logger().warn(
                    f'overlay publish failed: {exc}', throttle_duration_sec=2.0,
                )

        if self.publish_target and target_x is not None:
            self.target_pub.publish(Float32(data=float(target_x)))

        self.predictions_pub.publish(String(data=preds_json))

    def destroy_node(self):
        self._stop_event.set()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoboflowLaneNode()
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
