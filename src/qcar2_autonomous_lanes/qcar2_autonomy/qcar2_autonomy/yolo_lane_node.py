#!/usr/bin/env python3
"""YOLOv8-seg lane instance segmentation for the QCar2 pipeline.

Loads a custom-trained Ultralytics YOLO checkpoint (best.pt) and produces:

    /yolo_lanes/debug_image   - sensor_msgs/Image  per-instance colored mask
                                overlay plus traffic-light boxes and target_x
                                marker.
    /planning/validated_target_x  (optional, default on)
                              - std_msgs/Float32   the lane-center pixel x,
                                derived from the detected lane masks at the
                                bottom of the image. Drops straight into the
                                existing pid_lane_follower_node.
    /yolo_lanes/traffic_light - std_msgs/String    "RED" / "YELLOW" / "GREEN"
                                / "NONE"  (best-effort: dominant pixel hue of
                                the highest-confidence traffic_light box).

Model the user trained outputs class names:
    0: 'lane2'         - each instance is one detected lane LINE (per-pixel mask)
    1: 'traffic_light' - bounding box around a traffic light

Multiple lane2 instances per frame are common (one per visible lane line).
We sort them left-to-right by mask centroid x, then take the two whose x
sits closest to either side of the camera center as "current-lane left/right
edge" and publish the midpoint as target_x.
"""
from pathlib import Path
import time

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist  # noqa: F401  (kept for future direct control)
from qcar2_autonomy.image_msg_utils import bgr_to_image_msg
from qcar2_autonomy.image_msg_utils import image_msg_to_bgr
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


# BGR palette: distinct colors per lane instance, cycled.
LANE_PALETTE = [
    (0, 255, 255),    # yellow
    (0, 255, 0),      # green
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 0, 255),      # red
    (255, 128, 0),    # orange-ish
]
TRAFFIC_LIGHT_COLOR = (0, 200, 255)  # orange-ish bbox


class YoloLaneNode(Node):
    """YOLOv8-seg lane instance segmenter -> colored overlay + target_x."""

    def __init__(self):
        super().__init__('yolo_lane_node')

        self.declare_parameter('model_path', self.default_model_path())
        self.declare_parameter('image_topic', '/qcar2/front_camera/image')
        self.declare_parameter('debug_image_topic', '/yolo_lanes/debug_image')
        self.declare_parameter('target_topic', '/planning/validated_target_x')
        self.declare_parameter('traffic_light_topic', '/yolo_lanes/traffic_light')
        self.declare_parameter('publish_target', True)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('infer_rate_hz', 5.0)
        self.declare_parameter('lane_class_name', 'lane2')
        self.declare_parameter('traffic_light_class_name', 'traffic_light')
        # Estimated lane width (pixels in the original frame). Used when only
        # one lane line is visible to project a fallback target.
        self.declare_parameter('lane_width_px', 420.0)
        # Mask blend opacity for the colored overlay (0..1).
        self.declare_parameter('mask_alpha', 0.45)

        model_path = str(self.get_parameter('model_path').value)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'YOLO weights not found at {model_path}')

        # Import ultralytics lazily so the rest of the package stays importable
        # on machines without it.
        from ultralytics import YOLO  # noqa: E402
        self.get_logger().info(f'loading YOLO weights from {model_path}...')
        t0 = time.monotonic()
        self.model = YOLO(model_path)
        self.get_logger().info(
            f'YOLO ready in {time.monotonic() - t0:.2f}s   task={self.model.task}   '
            f'names={self.model.names}'
        )

        # Resolve class IDs from configured names (defensive: handles training
        # runs where the class order changes).
        self.lane_class_id = self._find_class_id(
            str(self.get_parameter('lane_class_name').value)
        )
        self.tl_class_id = self._find_class_id(
            str(self.get_parameter('traffic_light_class_name').value)
        )

        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.device = str(self.get_parameter('device').value)
        self.publish_target = bool(self.get_parameter('publish_target').value)
        self.lane_width_px = float(self.get_parameter('lane_width_px').value)
        self.mask_alpha = float(self.get_parameter('mask_alpha').value)

        self.latest_bgr = None
        self.latest_header = None

        self.image_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, 10
        )
        self.target_pub = self.create_publisher(
            Float32, self.get_parameter('target_topic').value, 10
        )
        self.tl_pub = self.create_publisher(
            String, self.get_parameter('traffic_light_topic').value, 10
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image,
            10,
        )

        rate = float(self.get_parameter('infer_rate_hz').value)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'YOLO lane node ready: image={self.get_parameter("image_topic").value} '
            f'-> overlay={self.get_parameter("debug_image_topic").value}, '
            f'target={self.get_parameter("target_topic").value} '
            f'(publish={self.publish_target}), '
            f'conf={self.conf_threshold} imgsz={self.imgsz} device={self.device}'
        )

    @staticmethod
    def default_model_path():
        """Return the best.pt path from source or installed package share."""
        source_candidate = (
            Path(__file__).resolve().parents[1] / 'weights' / 'best.pt'
        )
        candidates = [source_candidate]
        try:
            share_dir = Path(get_package_share_directory('qcar2_autonomy'))
            candidates.append(share_dir / 'weights' / 'best.pt')
        except PackageNotFoundError:
            pass

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return str(source_candidate)

    def _find_class_id(self, target_name):
        for cid, name in self.model.names.items():
            if name == target_name:
                return int(cid)
        self.get_logger().warn(
            f'class "{target_name}" not in model.names={self.model.names}'
        )
        return None

    def _on_image(self, msg: Image):
        try:
            self.latest_bgr = image_msg_to_bgr(msg)
            self.latest_header = msg.header
        except (cv2.error, ValueError) as exc:
            self.get_logger().warn(
                f'image convert failed: {exc}',
                throttle_duration_sec=2.0,
            )

    def _tick(self):
        if self.latest_bgr is None:
            return
        original = self.latest_bgr.copy()
        h0, w0 = original.shape[:2]

        try:
            t0 = time.monotonic()
            results = self.model.predict(
                original,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            infer_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'YOLO inference failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        result = results[0]  # single image

        # Collect lane instances and traffic-light boxes from the result.
        lane_instances = []   # list of dicts: {mask, conf, cx}
        traffic_lights = []   # list of dicts: {box, conf}

        if result.masks is not None and result.boxes is not None:
            # Masks come back at model imgsz; ultralytics .data is already
            # in the original image shape if we passed the raw frame.
            masks = result.masks.data.cpu().numpy() > 0.5  # (N, H, W)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for i, cls_id in enumerate(classes):
                if cls_id == self.lane_class_id:
                    mask = masks[i]
                    # Resize mask to original frame if shapes differ
                    if mask.shape != (h0, w0):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (w0, h0),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    cx = self._mask_bottom_centroid_x(mask, h0)
                    lane_instances.append({
                        'mask': mask, 'conf': float(confs[i]), 'cx': cx,
                    })
                elif cls_id == self.tl_class_id:
                    traffic_lights.append({
                        'box': boxes[i], 'conf': float(confs[i]),
                    })

        # Sort lane instances left-to-right by their bottom-centroid x.
        lane_instances.sort(key=lambda d: (d['cx'] is None, d['cx'] or 0.0))

        # Build the colored overlay.
        overlay = original.copy()
        for idx, lane in enumerate(lane_instances):
            color = LANE_PALETTE[idx % len(LANE_PALETTE)]
            self._paint_mask(overlay, lane['mask'], color)
        # Alpha blend so the road texture is still visible under the masks.
        blended = cv2.addWeighted(overlay, self.mask_alpha,
                                  original, 1.0 - self.mask_alpha, 0.0)

        # Draw traffic-light boxes (no mask blend - just rectangles + label).
        tl_color = self._dominant_traffic_light_color(original, traffic_lights)
        for tl in traffic_lights:
            x1, y1, x2, y2 = tl['box'].astype(int)
            cv2.rectangle(blended, (x1, y1), (x2, y2), TRAFFIC_LIGHT_COLOR, 2)
            cv2.putText(
                blended, f'tl {tl["conf"]:.2f}', (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TRAFFIC_LIGHT_COLOR, 1,
            )

        # Compute & draw target_x.
        target_x = self._compute_target_x(lane_instances, w0)
        if target_x is not None:
            cv2.line(blended, (int(target_x), 0), (int(target_x), h0),
                     (255, 0, 255), 2)
        cv2.line(blended, (w0 // 2, 0), (w0 // 2, h0), (0, 0, 255), 1)

        # HUD
        tx_str = f'{target_x:.1f}' if target_x is not None else 'none'
        hud = (f'infer: {infer_ms:.0f}ms   lanes: {len(lane_instances)}   '
               f'tl: {len(traffic_lights)} ({tl_color})   target_x={tx_str}')
        cv2.putText(blended, hud, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(blended, hud, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        for i, lane in enumerate(lane_instances):
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            cx_str = f'{lane["cx"]:.0f}' if lane['cx'] is not None else '?'
            cv2.putText(blended, f'lane{i} cx={cx_str} conf={lane["conf"]:.2f}',
                        (10, 22 + 20 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        try:
            msg = bgr_to_image_msg(blended, header=self.latest_header)
            self.image_pub.publish(msg)
        except (cv2.error, ValueError) as exc:
            self.get_logger().warn(
                f'debug image publish failed: {exc}',
                throttle_duration_sec=2.0,
            )

        if self.publish_target and target_x is not None:
            self.target_pub.publish(Float32(data=float(target_x)))

        self.tl_pub.publish(String(data=tl_color))

    def _mask_bottom_centroid_x(self, mask, h):
        """Mean x of the bottom 25% of the mask's white pixels.

        Using the bottom-only centroid makes the ordering robust against
        long curving masks that wander toward the horizon.
        """
        if not mask.any():
            return None
        y_lo = int(0.75 * h)
        cropped = mask[y_lo:, :]
        if cropped.any():
            ys, xs = np.where(cropped)
            return float(xs.mean())
        ys, xs = np.where(mask)
        return float(xs.mean())

    @staticmethod
    def _paint_mask(canvas, mask, color):
        canvas[mask] = color

    def _compute_target_x(self, lane_instances, w):
        """Lane center: midpoint of the two masks bracketing the camera center.

        Falls back to single-line offset if only one lane mask is detected.
        """
        if not self.publish_target:
            return None
        cam_center = 0.5 * w
        valid = [d for d in lane_instances if d['cx'] is not None]
        if not valid:
            return None
        if len(valid) == 1:
            cx = valid[0]['cx']
            half = 0.5 * self.lane_width_px
            return max(0.0, min(float(w - 1),
                       cx - half if cx > cam_center else cx + half))
        # Two or more lanes: pick the one closest to center from each side.
        lefts = [d for d in valid if d['cx'] < cam_center]
        rights = [d for d in valid if d['cx'] >= cam_center]
        if lefts and rights:
            left_x = max(d['cx'] for d in lefts)
            right_x = min(d['cx'] for d in rights)
            return 0.5 * (left_x + right_x)
        # All on one side: project from the nearest one.
        cx_nearest = min(valid, key=lambda d: abs(d['cx'] - cam_center))['cx']
        half = 0.5 * self.lane_width_px
        return max(0.0, min(float(w - 1),
                   cx_nearest - half if cx_nearest > cam_center else cx_nearest + half))

    @staticmethod
    def _dominant_traffic_light_color(bgr, traffic_lights):
        """Crude red/yellow/green classifier for the highest-confidence box."""
        if not traffic_lights:
            return 'NONE'
        tl = max(traffic_lights, key=lambda t: t['conf'])
        x1, y1, x2, y2 = tl['box'].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 'NONE'
        crop = bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        bright = (s > 80) & (v > 80)
        if not bright.any():
            return 'NONE'
        # Hue in OpenCV is [0, 180]. Red wraps around 0/180.
        hue = h[bright]
        red = ((hue < 15) | (hue > 165)).sum()
        yellow = ((hue >= 15) & (hue < 35)).sum()
        green = ((hue >= 35) & (hue < 85)).sum()
        top = max(('RED', red), ('YELLOW', yellow), ('GREEN', green),
                  key=lambda kv: kv[1])
        return top[0] if top[1] > 0 else 'NONE'


def main(args=None):
    rclpy.init(args=args)
    node = YoloLaneNode()
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
