#!/usr/bin/env python3
"""Run a Roboflow-hosted model locally with cached weights.

Roboflow blocks raw .pt downloads ("403 Not authorized to download this model
in pt format") so the only supported way to get a Roboflow project running
offline is `inference.get_model(model_id, api_key=...)`. That call:

    1. Hits Roboflow once to fetch the model's serialized artifacts
       (TorchScript / ONNX / etc., depending on architecture).
    2. Caches them under MODEL_CACHE_DIR (we override the default here).
    3. Returns a Python object whose ``.infer(image)`` runs locally.

After the first run, the cache is populated and subsequent runs work
fully offline (use ``--no-http-fallback`` to enforce that).

Usage examples:
    # one-shot on a single image, caches into ./weights/roboflow_cache:
    ROBOFLOW_API_KEY=... python3 roboflow_local_inference.py --image my_frame.jpg

    # batch a folder, write annotated images + JSON to roboflow_outputs/:
    ROBOFLOW_API_KEY=... python3 roboflow_local_inference.py --input-dir frames/

    # subsequent runs purely offline:
    python3 roboflow_local_inference.py --image my_frame.jpg --no-http-fallback
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

DEFAULT_MODEL_ID = "car-track-nkz9u/3"
DEFAULT_OUTPUT_DIR = "roboflow_outputs"
DEFAULT_CACHE_DIR = "weights/roboflow_cache"
DEFAULT_CONFIDENCE = 0.5

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class _C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def info(msg: str) -> None:
    print(f"{_C.GREEN}[i]{_C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{_C.YELLOW}[!]{_C.RESET} {msg}")


def error(msg: str) -> None:
    print(f"{_C.RED}[x]{_C.RESET} {msg}", file=sys.stderr)


def debug(msg: str, *, enabled: bool) -> None:
    if enabled:
        print(f"{_C.DIM}[debug] {msg}{_C.RESET}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help=f"Roboflow model_id (default: {DEFAULT_MODEL_ID})")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", type=Path, help="path to a single input image")
    src.add_argument("--input-dir", type=Path,
                     help="folder of input images (recursive, common suffixes)")
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                   help=f"folder for annotated images + JSON (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR),
                   help=f"local model artifact cache (default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"min confidence threshold (default: {DEFAULT_CONFIDENCE})")
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                   help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    p.add_argument("--no-http-fallback", action="store_true",
                   help="force offline mode (use only cached artifacts)")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p.parse_args()


def ensure_inputs(args: argparse.Namespace) -> List[Path]:
    if args.image is None and args.input_dir is None:
        raise SystemExit("must pass --image or --input-dir")
    if args.image:
        if not args.image.is_file():
            raise SystemExit(f"image not found: {args.image}")
        return [args.image]
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory not found: {args.input_dir}")
    paths = sorted(
        p for p in args.input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"no images under {args.input_dir} "
                         f"(suffixes: {sorted(IMG_SUFFIXES)})")
    return paths


def prepare_environment(args: argparse.Namespace) -> Path:
    """Configure cache + offline behaviour BEFORE importing inference."""
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Two different cache vars depending on which model loader fires:
    #   MODEL_CACHE_DIR        -> used by the `inference` package (metadata)
    #   INFERENCE_HOME         -> used by `inference-models` autoloader
    #                              (this is where the actual weights go)
    # Setting both keeps everything in one persistent location. The defaults
    # for both are "/tmp/cache" which gets wiped on reboot.
    os.environ["MODEL_CACHE_DIR"] = str(cache_dir)
    os.environ["INFERENCE_HOME"] = str(cache_dir)
    os.environ["INFERENCE_MODEL_CACHE_DIR"] = str(cache_dir)
    os.environ["INFERENCE_MODEL_DIR"] = str(cache_dir)
    if args.no_http_fallback:
        os.environ["OFFLINE_MODE"] = "true"
        os.environ["ROBOFLOW_DISABLE_API"] = "true"
        os.environ["INFERENCE_DISABLE_HTTP_FALLBACK"] = "1"
        os.environ["DISABLE_INFERENCE_API_FALLBACK"] = "True"
    if args.debug:
        os.environ["INFERENCE_LOG_LEVEL"] = "DEBUG"
    if args.api_key:
        os.environ["ROBOFLOW_API_KEY"] = args.api_key
    debug(f"MODEL_CACHE_DIR={cache_dir}", enabled=args.debug)
    debug(f"OFFLINE_MODE={os.environ.get('OFFLINE_MODE', 'false')}",
          enabled=args.debug)
    return cache_dir


def load_model(args: argparse.Namespace):
    try:
        from inference import get_model  # type: ignore
    except ImportError as exc:
        error("the `inference` package is not installed.")
        error("  pip install inference  (see installation notes in script header)")
        raise SystemExit(1) from exc

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    info(f"loading model: {args.model_id}")
    if api_key:
        debug(f"using ROBOFLOW_API_KEY=***{api_key[-4:]}", enabled=args.debug)
    else:
        warn("no API key set. First-time downloads will fail without one.")
    t0 = time.monotonic()
    try:
        model = get_model(model_id=args.model_id, api_key=api_key or None)
    except Exception as exc:
        error(f"failed to load model: {exc}")
        if args.no_http_fallback:
            error("offline mode is on but the cache is empty.")
            error(f"  -> remove --no-http-fallback once to populate "
                  f"{args.cache_dir}")
        raise SystemExit(2) from exc
    info(f"model ready in {time.monotonic() - t0:.2f}s")
    return model


def _color_for_class(cls: str) -> tuple:
    palette = [
        (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
        (0, 0, 255), (255, 128, 0), (128, 0, 255),
    ]
    return palette[hash(cls) % len(palette)]


def _pred_get(pred, key: str, default=None):
    """Unified accessor for dict-style or object-style prediction entries.

    Roboflow's newer responses use ``class_name`` instead of ``class`` for
    the label string, so ``class`` lookups transparently fall back to
    ``class_name`` (and ``cls``) in either dict or object form.
    """
    aliases = {"class": ("class", "class_name", "cls")}
    keys_to_try = aliases.get(key, (key,))
    if isinstance(pred, dict):
        for k in keys_to_try:
            if k in pred:
                return pred[k]
        return default
    for k in keys_to_try:
        if hasattr(pred, k):
            return getattr(pred, k)
    return default


def _pred_points(pred):
    """Return polygon points as list of (x, y) ints, or None."""
    pts = _pred_get(pred, "points")
    if not pts:
        return None
    out = []
    for pt in pts:
        if isinstance(pt, dict):
            out.append((int(round(pt["x"])), int(round(pt["y"]))))
        else:
            out.append((int(round(pt.x)), int(round(pt.y))))
    return out


def draw_annotations(image, predictions: list, *, debug_mode: bool):
    """Draw bboxes + labels + polygon overlay on a BGR image."""
    import cv2
    import numpy as np

    for pred in predictions:
        cls = str(_pred_get(pred, "class", "?"))
        conf = float(_pred_get(pred, "confidence", 0.0) or 0.0)
        cx = _pred_get(pred, "x")
        cy = _pred_get(pred, "y")
        w = _pred_get(pred, "width")
        h = _pred_get(pred, "height")
        color = _color_for_class(cls)

        # Polygon (segmentation) — draw first so the bbox sits on top
        poly_pts = _pred_points(pred)
        if poly_pts:
            poly = np.array(poly_pts, dtype=np.int32)
            overlay = image.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0, dst=image)
            cv2.polylines(image, [poly], True, color, 2)

        # Bounding box
        if cx is not None and cy is not None and w is not None and h is not None:
            x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
            x2, y2 = int(round(cx + w / 2)), int(round(cy + h / 2))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, max(0, y1 - th - bl - 4)),
                          (x1 + tw + 6, y1), color, -1)
            cv2.putText(image, label, (x1 + 3, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        if debug_mode:
            n_poly = len(poly_pts) if poly_pts else 0
            debug(f"  pred class={cls} conf={conf:.3f} bbox=({cx},{cy},{w},{h}) "
                  f"polypts={n_poly}", enabled=True)
    return image


def normalize_response(response) -> tuple[list, dict]:
    """Convert inference response (object or dict) into (predictions, raw_dict)."""
    if isinstance(response, list):
        response = response[0] if response else {}
    if hasattr(response, "model_dump"):
        raw = response.model_dump()
    elif hasattr(response, "dict"):
        raw = response.dict(by_alias=True, exclude_none=True)
    elif isinstance(response, dict):
        raw = response
    else:
        raise TypeError(f"unexpected response type: {type(response).__name__}")
    preds_raw = raw.get("predictions", []) or []
    # Predictions may be a list of objects; coerce each to dict for JSON dump.
    preds_norm = []
    for p in preds_raw:
        if isinstance(p, dict):
            preds_norm.append(p)
        elif hasattr(p, "model_dump"):
            preds_norm.append(p.model_dump())
        elif hasattr(p, "dict"):
            preds_norm.append(p.dict(exclude_none=True))
        else:
            preds_norm.append({
                "class": getattr(p, "class_name", getattr(p, "cls", "?")),
                "confidence": getattr(p, "confidence", 0.0),
                "x": getattr(p, "x", None), "y": getattr(p, "y", None),
                "width": getattr(p, "width", None), "height": getattr(p, "height", None),
            })
    raw["predictions"] = preds_norm
    return preds_norm, raw


def process_one(model, image_path: Path, output_dir: Path,
                confidence: float, debug_mode: bool) -> dict:
    """Run inference on one image, save annotated PNG + JSON, return summary."""
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {image_path}")

    t0 = time.monotonic()
    response = model.infer(image, confidence=confidence)
    infer_ms = (time.monotonic() - t0) * 1000.0

    preds, raw = normalize_response(response)
    annotated = draw_annotations(image.copy(), preds, debug_mode=debug_mode)

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"{image_path.name}   preds={len(preds)}   infer={infer_ms:.0f}ms",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        cv2.LINE_AA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    annotated_path = output_dir / f"{stem}_annotated.png"
    json_path = output_dir / f"{stem}_predictions.json"
    cv2.imwrite(str(annotated_path), annotated)
    json_path.write_text(json.dumps(raw, indent=2, default=str))

    info(f"{image_path.name}: {len(preds)} pred(s) in {infer_ms:.0f}ms "
         f"-> {annotated_path}")
    return {
        "image": str(image_path),
        "annotated": str(annotated_path),
        "json": str(json_path),
        "n_predictions": len(preds),
        "infer_ms": infer_ms,
    }


def main() -> int:
    args = parse_args()
    inputs = ensure_inputs(args)
    cache_dir = prepare_environment(args)
    model = load_model(args)

    info(f"cache: {cache_dir}")
    info(f"processing {len(inputs)} image(s) at conf >= {args.confidence:.2f}")
    summaries: List[dict] = []
    failed: List[str] = []
    for path in inputs:
        try:
            summaries.append(process_one(model, path, args.output_dir,
                                         args.confidence, args.debug))
        except Exception as exc:
            error(f"{path}: {exc}")
            failed.append(str(path))

    print()
    info(f"done. {len(summaries)} succeeded, {len(failed)} failed.")
    if summaries:
        total_preds = sum(s["n_predictions"] for s in summaries)
        avg_ms = sum(s["infer_ms"] for s in summaries) / len(summaries)
        info(f"total predictions: {total_preds}, avg inference: {avg_ms:.0f}ms")
    if failed:
        warn(f"failed inputs: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
