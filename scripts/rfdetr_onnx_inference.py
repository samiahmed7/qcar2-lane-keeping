#!/usr/bin/env python3
"""Self-contained RF-DETR-Seg ONNX inference.

Single dependency footprint: ``onnxruntime + opencv-python + numpy``.
No Roboflow account, no `inference` package, no cache folder. Just three
files:

    weights/car_track_v3_lane.onnx        ~130 MB  - the model
    weights/car_track_v3_lane.classes.txt ~50 B    - class names per row
    scripts/rfdetr_onnx_inference.py      ~6 KB    - this script

Copy those three anywhere -> run -> done.

Model details (auto-discovered from the .onnx graph):
    input:  [1, 3, 432, 432] float32       NCHW, ImageNet-normalised RGB
    dets:   [1, 200, 4]                    cx, cy, w, h (normalized to 0..1)
    labels: [1, 200, num_classes]          per-class scores (sigmoid output)
    masks:  [1, 200, 108, 108]             mask logits (sigmoid -> binarise)

Usage:
    python3 rfdetr_onnx_inference.py --image my_frame.jpg
    python3 rfdetr_onnx_inference.py --input-dir frames/ --output-dir out/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort


# Model contract (constants because the ONNX baked them in).
INPUT_SIZE = 432
MASK_SIZE = 108

# DINOv2-backbone preprocessing: ImageNet mean/std on /255 RGB.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class index 0 in this model is the background placeholder
# ("background_class83422" per Roboflow's inference_config). Ignore it.
SKIP_CLASS_IDX = 0


# ---------------- io helpers ----------------

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_onnx = here.parent / "weights" / "car_track_v3_lane.onnx"
    default_classes = here.parent / "weights" / "car_track_v3_lane.classes.txt"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="path to one image")
    src.add_argument("--input-dir", type=Path, help="folder of images (recursive)")
    p.add_argument("--onnx", type=Path, default=default_onnx,
                   help=f"ONNX weights (default: {default_onnx})")
    p.add_argument("--classes", type=Path, default=default_classes,
                   help=f"class-name file, one per line (default: {default_classes})")
    p.add_argument("--output-dir", type=Path, default=Path("rfdetr_outputs"))
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.5,
                   help="mask logit -> binarisation threshold (default 0.5)")
    p.add_argument("--provider", default="CPUExecutionProvider",
                   help="onnxruntime provider (CUDAExecutionProvider for GPU)")
    return p.parse_args()


def list_inputs(args: argparse.Namespace) -> List[Path]:
    if args.image:
        if not args.image.is_file():
            raise SystemExit(f"image not found: {args.image}")
        return [args.image]
    if not args.input_dir.is_dir():
        raise SystemExit(f"input-dir not found: {args.input_dir}")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in args.input_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in suffixes)


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"classes file missing: {path}")
    raw = path.read_text().strip()
    # Robust to both newline-separated and space-separated formats.
    names = [t for t in raw.replace("\n", " ").split() if t]
    if not names:
        raise SystemExit(f"no class names parsed from {path}")
    return names


# ---------------- pre/post processing ----------------

def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """BGR uint8 -> NCHW float32 ImageNet-normalised at 432x432.

    Returns:
        tensor          shape (1, 3, 432, 432) float32
        (orig_h, orig_w) so we can rescale outputs back to the original image
    """
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return chw, (h, w)


def postprocess(dets: np.ndarray, labels: np.ndarray, masks: np.ndarray,
                orig_shape: tuple[int, int],
                class_names: list[str],
                conf_thresh: float, mask_thresh: float) -> list[dict]:
    """Turn raw ONNX outputs into a list of {class, conf, bbox, mask} dicts.

    dets:   (200, 4)   normalized (cx, cy, w, h) in [0, 1] of the 432-input
    labels: (200, C)   per-class scores (sigmoid already applied for RF-DETR)
    masks:  (200, 108, 108)  mask logits
    """
    dets = dets[0]
    labels = labels[0]
    masks = masks[0]
    orig_h, orig_w = orig_shape

    # RF-DETR's "labels" output is raw logits per class. Apply sigmoid to get
    # per-class probabilities in [0, 1] before ranking. (Verified against
    # Roboflow's hosted prediction: logit 3.55 -> sigmoid -> 0.972 matches
    # their reported confidence of 0.97.)
    probs = 1.0 / (1.0 + np.exp(-labels.astype(np.float32)))
    if 0 <= SKIP_CLASS_IDX < probs.shape[1]:
        probs[:, SKIP_CLASS_IDX] = -np.inf
    best_cls = probs.argmax(axis=1)
    best_conf = probs[np.arange(probs.shape[0]), best_cls]

    keep = best_conf >= conf_thresh
    if not keep.any():
        return []

    results: list[dict] = []
    for i in np.where(keep)[0]:
        cls_idx = int(best_cls[i])
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"cls_{cls_idx}"
        conf = float(best_conf[i])

        # Box: normalized cx,cy,w,h -> pixel xyxy in the ORIGINAL image.
        cx, cy, bw, bh = dets[i].tolist()
        x1 = int(round((cx - bw / 2) * orig_w))
        y1 = int(round((cy - bh / 2) * orig_h))
        x2 = int(round((cx + bw / 2) * orig_w))
        y2 = int(round((cy + bh / 2) * orig_h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Mask: 108x108 logits -> sigmoid -> upscale to original size -> binarise.
        mlog = masks[i]
        mprob = 1.0 / (1.0 + np.exp(-mlog.astype(np.float32)))
        mfull = cv2.resize(mprob, (orig_w, orig_h),
                           interpolation=cv2.INTER_LINEAR)
        mbin = (mfull > mask_thresh).astype(np.uint8) * 255

        results.append({
            "class_index": cls_idx,
            "class": cls_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "mask": mbin,             # uint8 0/255 at original resolution
        })

    # Sort by confidence (descending) so the most-confident overlay sits on top.
    results.sort(key=lambda d: -d["confidence"])
    return results


# ---------------- visualisation ----------------

def color_for_class(name: str) -> tuple:
    palette = [
        (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
        (0, 0, 255), (255, 128, 0), (128, 0, 255),
    ]
    return palette[hash(name) % len(palette)]


def annotate(bgr: np.ndarray, predictions: list[dict]) -> np.ndarray:
    overlay = bgr.copy()
    for pred in predictions:
        color = color_for_class(pred["class"])
        # Find contour from the binary mask, then fill it.
        contours, _ = cv2.findContours(pred["mask"], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            fill = overlay.copy()
            cv2.fillPoly(fill, contours, color)
            cv2.addWeighted(fill, 0.35, overlay, 0.65, 0, dst=overlay)
            cv2.drawContours(overlay, contours, -1, color, 2)
        # bbox + label
        x1, y1, x2, y2 = pred["bbox_xyxy"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{pred['class']} {pred['confidence']:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - bl - 4)),
                      (x1 + tw + 6, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return overlay


# ---------------- main loop ----------------

def main() -> int:
    args = parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    classes = load_classes(args.classes)
    print(f"[i] classes: {classes}")
    print(f"[i] loading {args.onnx} ({args.onnx.stat().st_size/1e6:.1f} MB) "
          f"with provider={args.provider}")
    t0 = time.monotonic()
    sess = ort.InferenceSession(str(args.onnx), providers=[args.provider])
    print(f"[i] session ready in {time.monotonic() - t0:.2f}s")

    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    # We named the outputs in script docstring, but the third (mask) has the
    # graph-generated name "4647". Map by index.
    if len(out_names) != 3:
        raise SystemExit(f"unexpected ONNX output count: {len(out_names)}")

    inputs = list_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    for path in inputs:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[!] cannot read {path}")
            continue

        chw, orig_shape = preprocess(bgr)
        t_inf = time.monotonic()
        dets, labels, masks = sess.run(out_names, {input_name: chw})
        inf_ms = (time.monotonic() - t_inf) * 1000.0

        preds = postprocess(dets, labels, masks, orig_shape, classes,
                            args.confidence, args.mask_threshold)
        annotated = annotate(bgr, preds)

        # HUD bar
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(annotated,
                    f"{path.name}   preds={len(preds)}   infer={inf_ms:.0f}ms",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        stem = path.stem
        cv2.imwrite(str(args.output_dir / f"{stem}_annotated.png"), annotated)

        # Strip the binary mask from the JSON payload so it stays small.
        json_payload = [{k: v for k, v in p.items() if k != "mask"} for p in preds]
        (args.output_dir / f"{stem}_predictions.json").write_text(
            json.dumps(json_payload, indent=2))
        print(f"[i] {path.name}: {len(preds)} pred(s) in {inf_ms:.0f}ms "
              f"-> {args.output_dir / (stem + '_annotated.png')}")
        succeeded += 1

    print(f"[i] done. {succeeded} / {len(inputs)} succeeded.")
    return 0 if succeeded == len(inputs) else 1


if __name__ == "__main__":
    sys.exit(main())
