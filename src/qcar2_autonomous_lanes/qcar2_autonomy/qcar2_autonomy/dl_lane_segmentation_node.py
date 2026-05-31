#!/usr/bin/env python3
"""Ultralytics YOLO segmentation overlay for the QCar2 lane pipeline.

This node loads a trained YOLO segmentation checkpoint, extracts the lane mask
class, and publishes:

    /lane_segmentation/debug_image  (sensor_msgs/Image)  - colored mask overlay
    /lane_segmentation/lane_data    (Float32MultiArray)  - compact mask contours
    /planning/validated_target_x    (Float32)            - optional PID target

The control target is computed from lane mask instances near the bottom of the
image. With two or more lane-line masks, the node picks the pair that brackets
the camera center and publishes their midpoint. With one visible line, it falls
back to lane_width_px. Tune target_offset_px for a fixed image-space bias.
"""
from pathlib import Path
import time

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from qcar2_autonomy.image_msg_utils import bgr_to_image_msg
from qcar2_autonomy.image_msg_utils import image_msg_to_bgr
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray


LANE_COLOR = (0, 255, 0)             # BGR green
TRAFFIC_LIGHT_COLOR = (0, 140, 255)  # BGR orange
TARGET_COLOR = (255, 0, 255)         # BGR magenta


class DLLaneSegmentationNode(Node):
    """Run YOLO segmentation at a fixed rate and publish lane overlays."""

    def __init__(self):
        super().__init__('dl_lane_segmentation_node')

        self.declare_parameter('image_topic', '/qcar2/front_camera/image')
        self.declare_parameter(
            'debug_image_topic', '/lane_segmentation/debug_image'
        )
        self.declare_parameter(
            'lane_data_topic', '/lane_segmentation/lane_data'
        )
        self.declare_parameter('target_topic', '/planning/validated_target_x')
        self.declare_parameter('publish_target', True)

        self.declare_parameter('model_path', self.default_model_path())
        self.declare_parameter('lane_class_name', 'lane2')
        self.declare_parameter('traffic_light_class_name', 'traffic_light')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.50)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('infer_rate_hz', 10.0)

        self.declare_parameter('mask_alpha', 0.45)
        self.declare_parameter('target_roi_y_start_ratio', 0.55)
        self.declare_parameter('target_band_height_px', 80)
        self.declare_parameter('target_min_lane_pixels', 150)
        self.declare_parameter('lane_width_px', 420.0)
        self.declare_parameter('target_offset_px', 0.0)
        self.declare_parameter('max_lane_data_points', 200)

        self.model_path = str(self.get_parameter('model_path').value)
        self.lane_class_name = str(self.get_parameter('lane_class_name').value)
        self.traffic_light_class_name = str(
            self.get_parameter('traffic_light_class_name').value
        )
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.device = str(self.get_parameter('device').value).strip()
        self.publish_target = bool(self.get_parameter('publish_target').value)
        self.mask_alpha = float(self.get_parameter('mask_alpha').value)
        self.target_roi_y_start_ratio = float(
            self.get_parameter('target_roi_y_start_ratio').value
        )
        self.target_band_height_px = int(
            self.get_parameter('target_band_height_px').value
        )
        self.target_min_lane_pixels = int(
            self.get_parameter('target_min_lane_pixels').value
        )
        self.lane_width_px = float(self.get_parameter('lane_width_px').value)
        self.target_offset_px = float(self.get_parameter('target_offset_px').value)
        self.max_lane_data_points = int(
            self.get_parameter('max_lane_data_points').value
        )

        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f'YOLO weights not found at {self.model_path}')

        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                'ultralytics is required for dl_lane_segmentation_node. '
                'Install it with: pip install ultralytics'
            ) from exc

        self.get_logger().info(f'loading YOLO segmentation weights from {self.model_path}...')
        t0 = time.monotonic()
        self.model = YOLO(self.model_path)
        self.names = self.normalize_names(getattr(self.model, 'names', {}))
        self.lane_class_id = self.resolve_class_id(self.names, self.lane_class_name)
        self.traffic_light_class_id = self.resolve_class_id(
            self.names,
            self.traffic_light_class_name,
        )

        if self.lane_class_id is None:
            raise ValueError(
                f'lane class {self.lane_class_name!r} not found in model names: '
                f'{self.names}'
            )
        if self.traffic_light_class_id is None:
            self.get_logger().warn(
                f'traffic light class {self.traffic_light_class_name!r} not found '
                f'in model names: {self.names}'
            )

        task = getattr(self.model, 'task', 'unknown')
        if task != 'segment':
            self.get_logger().warn(
                f'loaded YOLO task={task!r}; expected a segmentation model'
            )

        self.get_logger().info(
            f'YOLO ready in {time.monotonic() - t0:.2f}s: '
            f'task={task}, names={self.names}, lane_class_id={self.lane_class_id}'
        )

        self.latest_bgr = None
        self.latest_header = None

        self.image_pub = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            10,
        )
        self.lane_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('lane_data_topic').value,
            10,
        )
        self.target_pub = self.create_publisher(
            Float32,
            self.get_parameter('target_topic').value,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image,
            10,
        )

        rate = max(0.1, float(self.get_parameter('infer_rate_hz').value))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'DL lane segmentation ready: '
            f'image={self.get_parameter("image_topic").value} -> '
            f'overlay={self.get_parameter("debug_image_topic").value} '
            f'@ {rate:.1f} Hz'
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

    @staticmethod
    def normalize_names(names):
        """Normalize Ultralytics class names to an int-keyed dictionary."""
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {int(class_id): str(name) for class_id, name in enumerate(names)}

    @staticmethod
    def resolve_class_id(names, class_name_or_id):
        """Resolve a model class id from either a class name or numeric string."""
        wanted = str(class_name_or_id).strip()
        if not wanted:
            return None

        try:
            numeric_id = int(wanted)
        except ValueError:
            numeric_id = None

        for class_id, name in names.items():
            if numeric_id is not None and int(class_id) == numeric_id:
                return int(class_id)
            if str(name).strip() == wanted:
                return int(class_id)
        return None

    @staticmethod
    def to_numpy(value):
        """Convert numpy/torch-like arrays to numpy without importing torch."""
        if value is None:
            return np.array([])
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        return np.asarray(value)

    @staticmethod
    def resize_mask(mask, width, height):
        """Convert a YOLO mask to a boolean image mask at camera resolution."""
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            mask = mask > 0.5
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return mask.astype(bool)

    @classmethod
    def extract_masks(
        cls,
        result,
        width,
        height,
        lane_class_id,
        traffic_light_class_id=None,
    ):
        """Extract combined lane/traffic masks and per-instance detections."""
        lane_mask = np.zeros((height, width), dtype=bool)
        traffic_light_mask = np.zeros((height, width), dtype=bool)
        detections = []

        boxes = getattr(result, 'boxes', None)
        masks = getattr(result, 'masks', None)
        if boxes is None or masks is None or getattr(masks, 'data', None) is None:
            return lane_mask, traffic_light_mask, detections

        classes = cls.to_numpy(getattr(boxes, 'cls', None)).astype(int)
        confidences = cls.to_numpy(getattr(boxes, 'conf', None)).astype(float)
        xyxy = cls.to_numpy(getattr(boxes, 'xyxy', None)).astype(float)
        mask_data = cls.to_numpy(masks.data)
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, :, :]

        count = min(len(mask_data), len(classes))
        for idx in range(count):
            class_id = int(classes[idx])
            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0
            box = xyxy[idx].tolist() if idx < len(xyxy) else [0.0, 0.0, 0.0, 0.0]
            mask = cls.resize_mask(mask_data[idx], width, height)

            if class_id == lane_class_id:
                lane_mask |= mask
                bottom_cx = cls.bottom_centroid_x(mask)
            elif traffic_light_class_id is not None and class_id == traffic_light_class_id:
                traffic_light_mask |= mask
                bottom_cx = None
            else:
                bottom_cx = None

            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'box': box,
                'bottom_cx': bottom_cx,
            })

        return lane_mask, traffic_light_mask, detections

    @staticmethod
    def bottom_centroid_x(mask, y_start_ratio=0.75):
        """Return mean x of mask pixels in the bottom part of the image."""
        mask = np.asarray(mask).astype(bool)
        if mask.ndim != 2 or not np.any(mask):
            return None

        height = mask.shape[0]
        y_start = int(round(height * float(y_start_ratio)))
        y_start = max(0, min(height - 1, y_start))
        cropped = mask[y_start:, :]
        if np.any(cropped):
            _, xs = np.where(cropped)
            return float(xs.mean())

        _, xs = np.where(mask)
        return float(xs.mean()) if xs.size else None

    @staticmethod
    def target_x_from_lane_instances(
        lane_instances,
        width,
        lane_width_px=420.0,
        target_offset_px=0.0,
    ):
        """Return target x from lane-line instance bottom centroids."""
        valid = [
            float(instance['bottom_cx'])
            for instance in lane_instances
            if instance.get('bottom_cx') is not None
        ]
        if not valid:
            return None

        camera_center = 0.5 * float(width)
        half_width = 0.5 * float(lane_width_px)

        if len(valid) == 1:
            cx = valid[0]
            target_x = cx - half_width if cx > camera_center else cx + half_width
        else:
            lefts = [cx for cx in valid if cx < camera_center]
            rights = [cx for cx in valid if cx >= camera_center]
            if lefts and rights:
                target_x = 0.5 * (max(lefts) + min(rights))
            else:
                cx = min(valid, key=lambda x: abs(x - camera_center))
                target_x = cx - half_width if cx > camera_center else cx + half_width

        target_x += float(target_offset_px)
        return max(0.0, min(float(width - 1), target_x))

    @staticmethod
    def target_x_from_lane_mask(
        lane_mask,
        min_pixels=150,
        y_start_ratio=0.55,
        band_height_px=80,
        target_offset_px=0.0,
    ):
        """Return a PID target x coordinate from the lowest visible lane mask."""
        mask = np.asarray(lane_mask).astype(bool)
        if mask.ndim != 2 or mask.size == 0:
            return None

        height, width = mask.shape
        if int(np.count_nonzero(mask)) < int(min_pixels):
            return None

        y_start = int(round(height * float(y_start_ratio)))
        y_start = max(0, min(height - 1, y_start))
        roi = mask[y_start:, :]
        ys, xs = np.where(roi)
        if xs.size >= int(min_pixels):
            ys = ys + y_start
        else:
            ys, xs = np.where(mask)
            if xs.size < int(min_pixels):
                return None

        bottom_y = int(np.max(ys))
        band_height = max(1, int(round(band_height_px)))
        band_start = max(0, bottom_y - band_height + 1)
        band = mask[band_start:bottom_y + 1, :]
        _, band_xs = np.where(band)
        if band_xs.size < int(min_pixels):
            band_xs = xs
        if band_xs.size == 0:
            return None

        x_left = float(np.percentile(band_xs, 5.0))
        x_right = float(np.percentile(band_xs, 95.0))
        target_x = 0.5 * (x_left + x_right) + float(target_offset_px)
        return max(0.0, min(float(width - 1), target_x))

    @staticmethod
    def mask_contour_data(mask, class_id, max_points):
        """Return [class_id, x, y, ...] sampled from mask contours."""
        if max_points <= 0 or not np.any(mask):
            return []

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        data = []
        point_count = 0
        for contour in contours:
            if point_count >= max_points:
                break
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(1.0, 0.01 * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            for x, y in approx:
                if point_count >= max_points:
                    break
                data.extend([float(class_id), float(x), float(y)])
                point_count += 1
        return data

    def _on_image(self, msg: Image):
        try:
            self.latest_bgr = image_msg_to_bgr(msg)
            self.latest_header = msg.header
        except (cv2.error, ValueError) as exc:
            self.get_logger().warn(
                f'image convert failed: {exc}',
                throttle_duration_sec=2.0,
            )

    def _predict(self, image_bgr):
        kwargs = {
            'source': image_bgr,
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'verbose': False,
        }
        if self.imgsz > 0:
            kwargs['imgsz'] = self.imgsz
        if self.device:
            kwargs['device'] = self.device
        return self.model.predict(**kwargs)[0]

    def _tick(self):
        if self.latest_bgr is None:
            return

        original = self.latest_bgr.copy()
        height, width = original.shape[:2]

        try:
            t0 = time.monotonic()
            result = self._predict(original)
            infer_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'YOLO inference failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        lane_mask, traffic_light_mask, detections = self.extract_masks(
            result,
            width,
            height,
            self.lane_class_id,
            self.traffic_light_class_id,
        )
        lane_instances = [
            detection
            for detection in detections
            if detection['class_id'] == self.lane_class_id
        ]

        target_x = None
        if self.publish_target:
            target_x = self.target_x_from_lane_instances(
                lane_instances,
                width,
                lane_width_px=self.lane_width_px,
                target_offset_px=self.target_offset_px,
            )
            if target_x is None:
                target_x = self.target_x_from_lane_mask(
                    lane_mask,
                    min_pixels=self.target_min_lane_pixels,
                    y_start_ratio=self.target_roi_y_start_ratio,
                    band_height_px=self.target_band_height_px,
                    target_offset_px=self.target_offset_px,
                )
            if target_x is not None:
                target_msg = Float32()
                target_msg.data = float(target_x)
                self.target_pub.publish(target_msg)

        overlay = self.render_overlay(
            original,
            lane_mask,
            traffic_light_mask,
            detections,
            target_x,
            infer_ms,
        )
        self.publish_debug_image(overlay)

        flat_data = self.mask_contour_data(
            lane_mask,
            self.lane_class_id,
            self.max_lane_data_points,
        )
        if traffic_light_mask.any() and self.traffic_light_class_id is not None:
            remaining_points = max(
                0,
                self.max_lane_data_points - len(flat_data) // 3,
            )
            flat_data.extend(
                self.mask_contour_data(
                    traffic_light_mask,
                    self.traffic_light_class_id,
                    remaining_points,
                )
            )
        if flat_data:
            data_msg = Float32MultiArray()
            data_msg.data = flat_data
            self.lane_pub.publish(data_msg)

    def render_overlay(
        self,
        original,
        lane_mask,
        traffic_light_mask,
        detections,
        target_x,
        infer_ms,
    ):
        overlay = original.copy()
        color_layer = np.zeros_like(original)
        color_layer[lane_mask] = LANE_COLOR
        color_layer[traffic_light_mask] = TRAFFIC_LIGHT_COLOR

        any_mask = lane_mask | traffic_light_mask
        if np.any(any_mask):
            alpha = max(0.0, min(1.0, self.mask_alpha))
            colored = cv2.addWeighted(original, 1.0 - alpha, color_layer, alpha, 0.0)
            overlay[any_mask] = colored[any_mask]

        for detection in detections:
            class_id = detection['class_id']
            name = self.names.get(class_id, str(class_id))
            confidence = detection['confidence']
            x1, y1, x2, y2 = [int(round(v)) for v in detection['box']]
            color = LANE_COLOR
            if class_id == self.traffic_light_class_id:
                color = TRAFFIC_LIGHT_COLOR
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                overlay,
                f'{name} {confidence:.2f}',
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        if target_x is not None:
            x = int(round(target_x))
            cv2.line(overlay, (x, 0), (x, original.shape[0]), TARGET_COLOR, 2)

        target_str = f'{target_x:.1f}' if target_x is not None else 'none'
        lane_pixels = int(np.count_nonzero(lane_mask))
        light_pixels = int(np.count_nonzero(traffic_light_mask))
        lines = [
            f'YOLO seg {infer_ms:.0f}ms  target_x={target_str}',
            f'lane pixels={lane_pixels}  traffic_light pixels={light_pixels}',
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (10, 22 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        return overlay

    def publish_debug_image(self, image_bgr):
        try:
            msg = bgr_to_image_msg(image_bgr, header=self.latest_header)
            self.image_pub.publish(msg)
        except (cv2.error, ValueError) as exc:
            self.get_logger().warn(
                f'debug image publish failed: {exc}',
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = DLLaneSegmentationNode()
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
