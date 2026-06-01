#!/usr/bin/env python3
"""GPU RF-DETR-Seg ONNX lane segmentation for the QCar2 pipeline.

Reads the standalone ONNX bundle (no Roboflow API key, no `inference` package)
that lives at:

    weights/car_track_v3_lane.onnx           (130 MB)
    weights/car_track_v3_lane.classes.txt    ('background_class83422 lane2 traffic_light')

On a GPU with onnxruntime-gpu + matching CUDA 12 runtime libs the model runs
at ~13 FPS (75 ms / frame). CPU fallback is ~1 FPS -- usable for diagnostics,
not for control.

Topology:
    /qcar2/front_camera/image (sensor_msgs/Image)
       -> rfdetr_onnx_lane_node (this)
            -> /rfdetr_lane/debug_image    (sensor_msgs/Image)  colored mask overlay
            -> /planning/validated_target_x (std_msgs/Float32)   feeds the PID

Inference runs on a background thread so the slow first frames don't gate
ROS callbacks. The publish timer always emits the most recent cached result.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from qcar2_autonomy.image_msg_utils import bgr_to_image_msg, image_msg_to_bgr
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


INPUT_SIZE = 432
MASK_SIZE = 108
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _color_for_class(name: str) -> tuple:
    palette = [(0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
               (0, 0, 255), (255, 128, 0)]
    return palette[hash(name) % len(palette)]


class RfdetrOnnxLaneNode(Node):
    """Drive the car from RF-DETR-Seg ONNX inference on the GPU."""

    def __init__(self):
        super().__init__('rfdetr_onnx_lane_node')

        # Model + classes
        ws = Path('/home/hammad/rosbot_ws')
        self.declare_parameter(
            'onnx_path',
            str(ws / 'weights' / 'car_track_v3_lane.onnx'))
        self.declare_parameter(
            'classes_path',
            str(ws / 'weights' / 'car_track_v3_lane.classes.txt'))
        self.declare_parameter('lane_class_name', 'lane2')
        self.declare_parameter('background_class_index', 0)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('mask_threshold', 0.5)
        self.declare_parameter('provider', 'CUDAExecutionProvider')

        # Topics
        self.declare_parameter('image_topic', '/qcar2/front_camera/image')
        self.declare_parameter('debug_image_topic', '/rfdetr_lane/debug_image')
        self.declare_parameter('target_topic', '/planning/validated_target_x')
        self.declare_parameter('predictions_topic', '/rfdetr_lane/predictions')
        self.declare_parameter('publish_target', True)

        # Control mapping
        # Bottom-band of the image used to read the lane center. Smaller =
        # closer to the car = more reactive, but noisier. Larger = smoother.
        self.declare_parameter('target_band_y_ratio', 0.70)
        # Sideways nudge on the computed target (pixels in the original frame).
        self.declare_parameter('target_offset_px', 0.0)
        self.declare_parameter('publish_rate_hz', 20.0)

        onnx_path = Path(str(self.get_parameter('onnx_path').value))
        classes_path = Path(str(self.get_parameter('classes_path').value))
        if not onnx_path.is_file():
            raise FileNotFoundError(f'ONNX not found: {onnx_path}')
        if not classes_path.is_file():
            raise FileNotFoundError(f'classes file not found: {classes_path}')

        self.lane_class_name = str(self.get_parameter('lane_class_name').value)
        self.background_class_index = int(
            self.get_parameter('background_class_index').value)
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value)
        self.mask_threshold = float(self.get_parameter('mask_threshold').value)
        self.target_band_y_ratio = float(
            self.get_parameter('target_band_y_ratio').value)
        self.target_offset_px = float(self.get_parameter('target_offset_px').value)
        self.publish_target = bool(self.get_parameter('publish_target').value)

        # Load model
        import onnxruntime as ort  # noqa: PLC0415
        self.classes = self._load_classes(classes_path)
        self.get_logger().info(
            f'classes={self.classes}, lane_class_name={self.lane_class_name}')

        providers = [str(self.get_parameter('provider').value)]
        if 'CPUExecutionProvider' not in providers:
            providers.append('CPUExecutionProvider')  # always have a fallback
        self.get_logger().info(
            f'loading {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB) '
            f'with providers={providers}')
        t0 = time.monotonic()
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.get_logger().info(
            f'session ready in {time.monotonic() - t0:.2f}s, '
            f'active providers={active}, inputs={self.input_name}, '
            f'outputs={self.output_names}')

        # Shared state between callback / worker / timer
        self._state_lock = threading.Lock()
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_overlay: Optional[np.ndarray] = None
        self.latest_target_x: Optional[float] = None
        self.latest_infer_ms: int = 0
        self.latest_n_preds: int = 0
        self._stop = threading.Event()

        # Pub/sub
        self.image_pub = self.create_publisher(
            Image, str(self.get_parameter('debug_image_topic').value), 10)
        self.target_pub = self.create_publisher(
            Float32, str(self.get_parameter('target_topic').value), 10)
        self.pred_pub = self.create_publisher(
            String, str(self.get_parameter('predictions_topic').value), 10)
        self.create_subscription(
            Image, str(self.get_parameter('image_topic').value),
            self._on_image, 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / rate, self._publish_tick)

        self.worker = threading.Thread(target=self._infer_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            'RF-DETR ONNX lane node up: '
            f'image={self.get_parameter("image_topic").value} -> '
            f'overlay={self.get_parameter("debug_image_topic").value}, '
            f'target={self.get_parameter("target_topic").value}, '
            f'conf>={self.confidence_threshold}')

    # ---- io ----

    @staticmethod
    def _load_classes(p: Path) -> list[str]:
        raw = p.read_text().strip()
        return [t for t in raw.replace('\n', ' ').split() if t]

    def _on_image(self, msg: Image):
        try:
            bgr = image_msg_to_bgr(msg)
        except (cv2.error, ValueError) as exc:
            self.get_logger().warn(
                f'image convert failed: {exc}', throttle_duration_sec=2.0)
            return
        with self._state_lock:
            self.latest_bgr = bgr

    # ---- worker ----

    def _infer_loop(self):
        while not self._stop.is_set():
            with self._state_lock:
                frame = (self.latest_bgr.copy()
                         if self.latest_bgr is not None else None)
            if frame is None:
                time.sleep(0.05)
                continue

            t0 = time.monotonic()
            chw, orig_shape = self._preprocess(frame)
            try:
                outs = self.session.run(self.output_names,
                                        {self.input_name: chw})
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'ONNX inference failed: {exc}',
                    throttle_duration_sec=2.0)
                time.sleep(0.1)
                continue
            infer_ms = int(round((time.monotonic() - t0) * 1000))

            dets, labels, masks = outs[0], outs[1], outs[2]
            preds = self._postprocess(dets, labels, masks, orig_shape)
            overlay, target_x = self._render(frame, preds, infer_ms)
            n_preds = len(preds)

            with self._state_lock:
                self.latest_overlay = overlay
                self.latest_target_x = target_x
                self.latest_infer_ms = infer_ms
                self.latest_n_preds = n_preds

    # ---- pre/post ----

    def _preprocess(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
        return chw, (h, w)

    def _postprocess(self, dets, labels, masks, orig_shape):
        dets = dets[0]
        labels = labels[0]
        masks = masks[0]
        orig_h, orig_w = orig_shape

        # Sigmoid the logits -> per-class probabilities. Mask background class
        # so it never wins the argmax.
        probs = 1.0 / (1.0 + np.exp(-labels.astype(np.float32)))
        if 0 <= self.background_class_index < probs.shape[1]:
            probs[:, self.background_class_index] = -np.inf
        best_cls = probs.argmax(axis=1)
        best_conf = probs[np.arange(probs.shape[0]), best_cls]
        keep_idx = np.where(best_conf >= self.confidence_threshold)[0]

        results = []
        for i in keep_idx:
            cls_idx = int(best_cls[i])
            cls_name = (self.classes[cls_idx]
                        if cls_idx < len(self.classes) else f'cls_{cls_idx}')
            conf = float(best_conf[i])
            cx, cy, bw, bh = dets[i].tolist()
            x1 = int(round((cx - bw / 2) * orig_w))
            y1 = int(round((cy - bh / 2) * orig_h))
            x2 = int(round((cx + bw / 2) * orig_w))
            y2 = int(round((cy + bh / 2) * orig_h))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            mlog = masks[i]
            mprob = 1.0 / (1.0 + np.exp(-mlog.astype(np.float32)))
            mfull = cv2.resize(mprob, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)
            mbin = (mfull > self.mask_threshold).astype(np.uint8) * 255
            results.append({
                'class': cls_name, 'class_index': cls_idx,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'mask': mbin,
            })
        results.sort(key=lambda d: -d['confidence'])
        return results

    def _render(self, bgr: np.ndarray, preds, infer_ms: int):
        h, w = bgr.shape[:2]
        overlay = bgr.copy()
        target_x = None

        # Pick the highest-confidence lane2 mask for control.
        lane_pred = next((p for p in preds
                          if p['class'] == self.lane_class_name), None)

        # Draw every prediction colored by class.
        for p in preds:
            color = _color_for_class(p['class'])
            contours, _ = cv2.findContours(p['mask'], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                fill = overlay.copy()
                cv2.fillPoly(fill, contours, color)
                cv2.addWeighted(fill, 0.35, overlay, 0.65, 0, dst=overlay)
                cv2.drawContours(overlay, contours, -1, color, 2)
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay,
                        f"{p['class']} {p['confidence']:.2f}",
                        (x1 + 3, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Compute target_x from the bottom band of the lane mask: the centroid
        # x of mask pixels in the bottom (1 - target_band_y_ratio) of the image
        # is roughly where the lane center is closest to the car.
        if lane_pred is not None:
            band_top = int(self.target_band_y_ratio * h)
            band = lane_pred['mask'][band_top:, :]
            if band.any():
                ys, xs = np.where(band > 0)
                target_x = float(np.mean(xs)) + self.target_offset_px
                target_x = max(0.0, min(float(w - 1), target_x))

        # Camera center + target marker
        cv2.line(overlay, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)
        if target_x is not None:
            cv2.line(overlay, (int(target_x), 0), (int(target_x), h),
                     (255, 0, 255), 2)

        # HUD
        active = self.session.get_providers()[0]
        n = len(preds)
        tx_s = f'{target_x:.1f}' if target_x is not None else 'none'
        hud = (f'RF-DETR ({active})   infer: {infer_ms}ms ({1000/max(1,infer_ms):.1f} FPS)'
               f'   preds: {n}   target_x={tx_s}')
        cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(overlay, hud, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return overlay, target_x

    # ---- publish ----

    def _publish_tick(self):
        with self._state_lock:
            overlay = self.latest_overlay
            target_x = self.latest_target_x
        if overlay is not None:
            try:
                self.image_pub.publish(bgr_to_image_msg(overlay))
            except (cv2.error, ValueError) as exc:
                self.get_logger().warn(
                    f'overlay publish failed: {exc}',
                    throttle_duration_sec=2.0)
        if self.publish_target and target_x is not None:
            self.target_pub.publish(Float32(data=float(target_x)))

    def destroy_node(self):
        self._stop.set()
        if hasattr(self, 'worker') and self.worker.is_alive():
            self.worker.join(timeout=5.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RfdetrOnnxLaneNode()
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
