#!/usr/bin/env python3
"""Attempt to load a Roboflow-cached weights.pth with Ultralytics YOLO.

Spoiler: for the specific checkpoint in this workspace (car-track-nkz9u/3),
this WILL FAIL because the model is RF-DETR-Seg-Medium, not YOLO. The two
architectures share no state_dict keys -- there is no transformation that
makes one load into the other. This script proves it systematically:

    1. Loads and deeply inspects the cached .pth.
    2. Reads sibling config files (inference_config.json, model_type.json,
       model_config.json, class_names.txt) to confirm the architecture.
    3. Detects whether the keys are YOLO-shaped, DETR-shaped, or something
       else, and reports the verdict before trying any conversion.
    4. Attempts every reasonable conversion strategy the user requested:
        a. raw_state_dict          - dump tensors as-is into a YOLO ckpt shell
        b. extract_model_entry     - if the .pth has {"model": tensor_dict}
        c. extract_state_dict      - if the .pth has {"state_dict": ...}
        d. wrap_with_meta          - emit a full {"model": <Ultralytics fake>,
                                                  "ema": None, "epoch": -1, ...}
       Each strategy reports success/failure with the exact reason.
    5. After saving each candidate, tries:
            from ultralytics import YOLO
            YOLO("best.pt")(test_image_or_zero_image)
       and reports the actual error.
    6. Recommends the working alternatives:
        - Run RF-DETR natively (already cached; we have a CLI for it).
        - Use the ONNX export that's already in shared-blobs (125 MB).
        - Retrain a YOLOv8/v11 segmentation model on the same dataset.

Usage:
    python3 convert_roboflow_to_ultralytics.py \\
        --pth weights/roboflow_cache/shared-blobs/cfc5d294220d4be9c9f110d66fedd15e \\
        --cache-dir weights/roboflow_cache \\
        --output-dir weights/converted \\
        --test-image /tmp/sim_for_roboflow.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


# Common YOLO key prefixes we expect when loading via Ultralytics.
YOLO_KEY_FINGERPRINTS = (
    "model.0.conv.weight",
    "model.22.dfl",
    "model.22.cv2",       # detect head
    "model.22.cv3",
    "model.22.proto",     # segment head
)

# Fingerprints that prove the checkpoint is NOT Ultralytics YOLO.
NON_YOLO_FINGERPRINTS = {
    "rfdetr": (
        "transformer.decoder.layers",
        "transformer.encoder.layers",
        "backbone.0.projector.stages",
        "level_embed",
        "sampling_offsets",
    ),
    "detr": (
        "transformer.encoder.layers",
        "class_embed.weight",
        "bbox_embed.layers",
    ),
    "huggingface_detr": (
        "model.backbone.conv_encoder",
        "model.encoder.layers",
        "model.decoder.layers",
    ),
}


@dataclass
class ConversionAttempt:
    name: str
    saved_path: Optional[Path] = None
    save_error: Optional[str] = None
    load_error: Optional[str] = None
    inference_error: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return (self.saved_path is not None
                and self.save_error is None
                and self.load_error is None
                and self.inference_error is None)


class _C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def info(msg: str) -> None:    print(f"{_C.GREEN}[i]{_C.RESET} {msg}")
def warn(msg: str) -> None:    print(f"{_C.YELLOW}[!]{_C.RESET} {msg}")
def error(msg: str) -> None:   print(f"{_C.RED}[x]{_C.RESET} {msg}", file=sys.stderr)
def heading(msg: str) -> None: print(f"\n{_C.BOLD}{_C.CYAN}== {msg} =={_C.RESET}")


# --------------------------------------------------------------------- #
# Step 1: load & deeply inspect the checkpoint
# --------------------------------------------------------------------- #

def load_checkpoint(pth_path: Path):
    if not pth_path.is_file():
        raise SystemExit(f"checkpoint not found: {pth_path}")
    info(f"loading {pth_path}")
    info(f"  size: {pth_path.stat().st_size / 1e6:.1f} MB")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    info(f"  top-level type: {type(ckpt).__name__}")
    return ckpt


def find_state_dict(ckpt) -> tuple[dict, str]:
    """Locate the actual tensor dict inside whatever wrapper the .pth has."""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], "ckpt['state_dict']"
        if "model" in ckpt:
            v = ckpt["model"]
            if isinstance(v, dict):
                # Could be a tensor dict OR a wrapped {state_dict: ...}
                if v and all(hasattr(t, "shape") for t in list(v.values())[:5]):
                    return v, "ckpt['model']"
                if "state_dict" in v:
                    return v["state_dict"], "ckpt['model']['state_dict']"
        # Bare state_dict?
        if ckpt and all(hasattr(t, "shape") for t in list(ckpt.values())[:5]):
            return ckpt, "ckpt (bare)"
    raise ValueError("could not locate a tensor state_dict in the checkpoint")


def inspect_state_dict(sd: dict) -> dict:
    """Print a summary + return a small dict with architecture flags."""
    keys = list(sd.keys())
    n = len(keys)
    total_params = sum(t.numel() for t in sd.values() if hasattr(t, "numel"))
    total_bytes = sum(t.element_size() * t.numel()
                      for t in sd.values() if hasattr(t, "numel"))
    info(f"  state_dict: {n} tensors, {total_params/1e6:.1f}M params, "
         f"{total_bytes/1e6:.1f} MB")

    # Print a representative sample of keys + shapes.
    print(f"{_C.DIM}  sample keys (first 5 / last 5):{_C.RESET}")
    for k in keys[:5] + (["    ..."] if n > 10 else []) + keys[-5:]:
        if k == "    ...":
            print(f"    {k}")
        else:
            t = sd[k]
            print(f"    {k}: {tuple(t.shape)}")

    # Architecture detection.
    arch = detect_architecture(sd)
    info(f"  detected architecture: {arch}")
    return {"arch": arch, "n_tensors": n, "n_params": total_params}


def detect_architecture(sd: dict) -> str:
    keys = list(sd.keys())
    text = " ".join(keys)
    # Count how many YOLO-fingerprint keys appear.
    yolo_hits = sum(1 for fp in YOLO_KEY_FINGERPRINTS if fp in text)
    if yolo_hits >= 2:
        return "ultralytics_yolo (likely)"
    for arch, fps in NON_YOLO_FINGERPRINTS.items():
        hits = sum(1 for fp in fps if fp in text)
        if hits >= 2:
            return f"{arch} (confirmed)"
    # Fall back to heuristic.
    if any("transformer" in k for k in keys):
        return "transformer (unknown variant)"
    if any(k.startswith("model.") and ".conv." in k for k in keys):
        return "yolo-like CNN (worth trying)"
    return "unknown"


# --------------------------------------------------------------------- #
# Step 2: read sibling config files for extra context
# --------------------------------------------------------------------- #

def read_sibling_configs(cache_dir: Path) -> dict:
    """Find the model_type.json + inference_config.json this project shipped."""
    summary: dict = {}
    candidates = [
        ("model_type.json",
         "shared-blobs/29f20a6e1d4b587155436c2d20d0b553"),
        ("inference_config.json",
         "shared-blobs/1364ed847420f898ff2afed206161716"),
        ("model_config.json", None),     # find via glob
        ("class_names.txt",
         "shared-blobs/f90a9bcb0e20e3321c29d23e4c2541f3"),
    ]
    info(f"scanning sibling configs under {cache_dir}")
    for name, default_blob in candidates:
        found = None
        if default_blob and (cache_dir / default_blob).is_file():
            found = cache_dir / default_blob
        else:
            matches = list(cache_dir.rglob(name))
            if matches:
                found = matches[0]
        if not found:
            continue
        try:
            if name.endswith(".json"):
                summary[name] = json.loads(found.read_text())
            else:
                summary[name] = found.read_text().strip()
        except (OSError, json.JSONDecodeError) as exc:
            summary[name] = f"<{exc}>"
            continue
        # Pretty-print compactly.
        print(f"  {name}: {found}")
        if isinstance(summary[name], dict):
            for k, v in list(summary[name].items())[:10]:
                preview = repr(v)[:80]
                print(f"    {k}: {preview}")
        else:
            preview = summary[name].replace("\n", " ")[:120]
            print(f"    {preview}")
    return summary


# --------------------------------------------------------------------- #
# Step 3: build candidate Ultralytics-shaped checkpoints
# --------------------------------------------------------------------- #

def _empty_ultralytics_shell(state_dict: dict) -> dict:
    """Wrap a state_dict in the dict shape Ultralytics writes to best.pt.

    Real Ultralytics best.pt files contain a serialized nn.Module under "model",
    plus metadata. Passing a bare state_dict via {"model": state_dict} matches
    the *shape* but the type check inside Ultralytics' attempt_load_one_weight
    looks for an actual Module. This wrapper is therefore best-effort -- it
    will only ever succeed if Ultralytics happens to have a fallback path
    for state_dict-style payloads (which it does NOT, last I checked).
    """
    return {
        "model": state_dict,
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {},
        "best_fitness": None,
        "fitness": None,
        "epoch": -1,
        "train_results": {},
        "license": "AGPL-3.0",
        "date": "2026-05-26",
        "version": "0.0",
    }


def try_strategy_raw_state_dict(sd: dict, out_dir: Path) -> ConversionAttempt:
    a = ConversionAttempt(name="raw_state_dict")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best_raw_state_dict.pt"
        torch.save(sd, out_path)
        a.saved_path = out_path
    except Exception as exc:  # noqa: BLE001
        a.save_error = repr(exc)
    return a


def try_strategy_extract_model_entry(ckpt, out_dir: Path) -> ConversionAttempt:
    a = ConversionAttempt(name="extract_model_entry")
    try:
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            a.save_error = "no 'model' key in top-level checkpoint"
            return a
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best_model_entry.pt"
        torch.save(ckpt["model"], out_path)
        a.saved_path = out_path
    except Exception as exc:  # noqa: BLE001
        a.save_error = repr(exc)
    return a


def try_strategy_extract_state_dict(ckpt, out_dir: Path) -> ConversionAttempt:
    a = ConversionAttempt(name="extract_state_dict")
    try:
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            a.save_error = "no 'state_dict' key in top-level checkpoint"
            return a
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best_state_dict.pt"
        torch.save(ckpt["state_dict"], out_path)
        a.saved_path = out_path
    except Exception as exc:  # noqa: BLE001
        a.save_error = repr(exc)
    return a


def try_strategy_wrap_with_meta(sd: dict, out_dir: Path) -> ConversionAttempt:
    a = ConversionAttempt(name="wrap_with_ultralytics_metadata")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best_wrapped.pt"
        torch.save(_empty_ultralytics_shell(sd), out_path)
        a.saved_path = out_path
    except Exception as exc:  # noqa: BLE001
        a.save_error = repr(exc)
    return a


def try_strategy_onnx_passthrough(cache_dir: Path, out_dir: Path) -> ConversionAttempt:
    """Copy the cached weights.onnx and try Ultralytics' generic ONNX loader.

    Ultralytics YOLO accepts ONNX paths for inference, but it inspects metadata
    embedded by their own exporter. A foreign ONNX may load partially or fail
    at the postprocess step. This strategy tells us the truth.
    """
    a = ConversionAttempt(name="onnx_passthrough")
    try:
        # The ONNX blob is a content-hashed file under shared-blobs.
        # 1. If models-cache symlinks exist + resolve, use them.
        # 2. Else, fall back to identifying ONNX by its file magic (b"ONNX" in
        #    the header) among the shared-blobs/ entries.
        src = None
        for sl in cache_dir.rglob("weights.onnx"):
            try:
                resolved = sl.resolve(strict=True)
                if resolved.is_file():
                    src = resolved
                    break
            except (OSError, RuntimeError):
                continue
        if src is None:
            blobs_dir = cache_dir / "shared-blobs"
            if blobs_dir.is_dir():
                # File magic disambiguation:
                #   .pth (PyTorch torch.save) -> starts with PK\x03\x04 (ZIP)
                #   .onnx (protobuf ModelProto) -> NOT zip; first byte is
                #       a protobuf varint (typically \x08 for the ir_version
                #       field tag). Confirm by looking for "pytorch" or
                #       "onnx" producer string in the first 1 KB.
                for f in blobs_dir.iterdir():
                    if not f.is_file() or f.stat().st_size < 1_000_000:
                        continue
                    with open(f, "rb") as fh:
                        head = fh.read(1024)
                    if head.startswith(b"PK\x03\x04"):
                        continue  # PyTorch zip checkpoint
                    if head[:1] == b"\x08" and (b"pytorch" in head.lower()
                                                 or b"onnx" in head.lower()):
                        src = f
                        break
        if src is None:
            a.save_error = "no weights.onnx found (symlinks broken; no ONNX-magic blob)"
            return a
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best.onnx"
        import shutil
        shutil.copyfile(src, out_path)
        a.saved_path = out_path
        a.notes.append(f"copied {src} -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    except Exception as exc:  # noqa: BLE001
        a.save_error = repr(exc)
    return a


def try_strategy_native_rfdetr(args, test_image: Optional[Path],
                               out_dir: Path) -> ConversionAttempt:
    """Bypass Ultralytics: run the cached weights natively via Roboflow's
    `inference` package and produce an annotated image. Counts as a 'conversion'
    only in the sense that the user wanted to see the model used on an image."""
    a = ConversionAttempt(name="native_rfdetr_inference")
    if test_image is None or not test_image.is_file():
        a.save_error = f"no test image at {test_image}"
        return a
    try:
        import os
        os.environ.setdefault("INFERENCE_HOME", str(args.cache_dir.resolve()))
        os.environ.setdefault("MODEL_CACHE_DIR", str(args.cache_dir.resolve()))
        os.environ.setdefault(
            "ONNXRUNTIME_EXECUTION_PROVIDERS", '["CPUExecutionProvider"]')
        os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
        os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
        os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
        from inference import get_model  # type: ignore
        import cv2
        import numpy as np

        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        model_id = args.model_id
        model = get_model(model_id=model_id, api_key=api_key or None)
        img = cv2.imread(str(test_image))
        if img is None:
            a.save_error = f"could not read {test_image}"
            return a
        response = model.infer(img, confidence=0.25)
        if isinstance(response, list):
            response = response[0] if response else None
        if response is None:
            a.notes.append("no predictions returned")
        # Annotate
        overlay = img.copy()
        n_preds = 0
        if hasattr(response, "predictions"):
            for p in response.predictions:
                n_preds += 1
                pts = getattr(p, "points", None)
                cx = getattr(p, "x", None)
                cy = getattr(p, "y", None)
                w = getattr(p, "width", None)
                h = getattr(p, "height", None)
                cls = getattr(p, "class_name", getattr(p, "class", "?"))
                conf = float(getattr(p, "confidence", 0.0))
                if pts:
                    poly = np.array([[int(round(pt.x)), int(round(pt.y))] for pt in pts],
                                     dtype=np.int32)
                    cv2.fillPoly(overlay, [poly], (0, 255, 255))
                    cv2.polylines(overlay, [poly], True, (0, 200, 255), 2)
                if cx is not None and w is not None:
                    x1, y1 = int(cx - w/2), int(cy - h/2)
                    x2, y2 = int(cx + w/2), int(cy + h/2)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(overlay, f"{cls} {conf:.2f}", (x1, max(0, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"native_rfdetr_{test_image.stem}_annotated.png"
        cv2.imwrite(str(out_path), blended)
        a.saved_path = out_path
        a.notes.append(f"native RF-DETR produced {n_preds} prediction(s) -> {out_path}")
    except Exception as exc:  # noqa: BLE001
        a.save_error = repr(exc)
    return a


# --------------------------------------------------------------------- #
# Step 4: test each candidate by trying to load with Ultralytics
# --------------------------------------------------------------------- #

def test_with_ultralytics(attempt: ConversionAttempt,
                          test_image: Optional[Path]) -> None:
    if attempt.saved_path is None:
        return
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        attempt.load_error = f"ultralytics not installed: {exc}"
        return

    try:
        model = YOLO(str(attempt.saved_path))
    except Exception as exc:  # noqa: BLE001
        tb_short = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        attempt.load_error = tb_short
        return

    if test_image is None or not test_image.is_file():
        attempt.notes.append("YOLO() loaded ok, but no test image to infer on")
        return

    # The Roboflow RF-DETR-Seg-Medium ONNX has a fixed input size of 432x432
    # (from inference_config.json). Ultralytics defaults to 640. Retry on size
    # mismatch with the smaller imgsz so we get a fairer error message.
    for imgsz in (640, 432):
        try:
            _ = model(str(test_image), imgsz=imgsz, verbose=False)
            attempt.notes.append(f"inference succeeded at imgsz={imgsz}")
            return
        except Exception as exc:  # noqa: BLE001
            tb_short = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            attempt.inference_error = tb_short
            if "Got invalid dimensions" in tb_short:
                continue  # try next size
            return


# --------------------------------------------------------------------- #
# Step 5: pretty-print the verdict
# --------------------------------------------------------------------- #

def print_verdict(attempts: list[ConversionAttempt], arch: str) -> int:
    heading("Verdict")
    any_success = any(a.succeeded for a in attempts)
    for a in attempts:
        if a.succeeded:
            info(f"{_C.GREEN}SUCCESS{_C.RESET} {a.name}: {a.saved_path}")
            for note in a.notes:
                print(f"    note: {note}")
        else:
            print(f"{_C.RED}FAIL{_C.RESET}    {a.name}")
            if a.save_error:
                print(f"    save_error:      {a.save_error}")
            if a.load_error:
                print(f"    load_error:      {a.load_error}")
            if a.inference_error:
                print(f"    inference_error: {a.inference_error}")
            for note in a.notes:
                print(f"    note: {note}")

    print()
    if any_success:
        info("at least one candidate loaded with Ultralytics. Inspect manually "
             "before trusting downstream inference -- a load that succeeds with "
             "missing/unexpected keys may still produce garbage predictions.")
        return 0

    # All failed -- explain what to do next.
    heading("Why the conversion failed")
    if "rfdetr" in arch or "detr" in arch:
        warn("The cached checkpoint is RF-DETR (Roboflow's DINOv2-backbone "
             "transformer detector). It is a fundamentally different "
             "architecture from Ultralytics YOLO. No state_dict remapping "
             "exists -- the two models share zero parameter tensors.")
    elif "transformer" in arch:
        warn("The checkpoint is some transformer detector, not a YOLO CNN. "
             "Ultralytics has no path to load transformer weights.")
    else:
        warn(f"Architecture detected: {arch}. None of the Ultralytics "
             "fallbacks accept this layout.")

    heading("What actually works (ranked)")
    print(f"""  1. {_C.BOLD}Run the model with its NATIVE backend{_C.RESET}
     We already have this working in this repo:

         python3 scripts/roboflow_local_inference.py \\
             --image my_frame.jpg --no-http-fallback

     That uses the cached weights.pth via Roboflow's `inference` package
     (the only thing that knows how to instantiate RF-DETR from these
     tensors).

  2. {_C.BOLD}Use the ONNX export that's already cached{_C.RESET}
     The same model is also cached as weights.onnx (125 MB) under
     weights/roboflow_cache/shared-blobs/. Load it with onnxruntime or
     openvino -- it's standard ONNX, runs on CPU or GPU without
     ultralytics.

         import onnxruntime as ort
         session = ort.InferenceSession('.../weights.onnx',
                                        providers=['CPUExecutionProvider'])

  3. {_C.BOLD}Retrain a YOLOv8/v11-seg on your dataset{_C.RESET}
     If you genuinely need a YOLO-shaped best.pt (e.g. for a
     YOLO-specific deployment target like Quanser's Jetson stack),
     train fresh on the dataset you already have:

         pip install ultralytics
         yolo task=segment mode=train model=yolov8n-seg.pt \\
             data=/tmp/rf_car_track_v3/data.yaml epochs=100 imgsz=640

     This gives you a real {_C.BOLD}best.pt{_C.RESET} in Ultralytics
     format, with the same training data Roboflow had.
""")
    return 1


# --------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    default_pth = (
        "weights/roboflow_cache/shared-blobs/"
        "cfc5d294220d4be9c9f110d66fedd15e"
    )
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pth", type=Path, default=Path(default_pth),
                   help="path to the cached weights.pth (content-hashed blob)")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("weights/roboflow_cache"),
                   help="root of the Roboflow cache, used to find sibling configs")
    p.add_argument("--output-dir", type=Path,
                   default=Path("weights/converted"),
                   help="where the candidate best_*.pt files get written")
    p.add_argument("--test-image", type=Path,
                   default=Path("/tmp/sim_for_roboflow.jpg"),
                   help="optional test image to run inference on (any loaded model)")
    p.add_argument("--model-id", default="car-track-nkz9u/3",
                   help="Roboflow model ID for the native fallback")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    heading("Step 1 : load + inspect cached checkpoint")
    ckpt = load_checkpoint(args.pth)
    try:
        sd, sd_origin = find_state_dict(ckpt)
        info(f"  state_dict found at: {sd_origin}")
    except ValueError as exc:
        error(str(exc))
        return 2
    inspection = inspect_state_dict(sd)

    heading("Step 2 : sibling config files")
    cfg = read_sibling_configs(args.cache_dir)
    if "model_type.json" in cfg and isinstance(cfg["model_type.json"], dict):
        mt = cfg["model_type.json"].get("model_type", "?")
        tt = cfg["model_type.json"].get("project_task_type", "?")
        info(f"  declared model_type: {mt}   task: {tt}")

    heading("Step 3 : attempt conversion strategies")
    attempts: list[ConversionAttempt] = []
    attempts.append(try_strategy_raw_state_dict(sd, args.output_dir))
    attempts.append(try_strategy_extract_model_entry(ckpt, args.output_dir))
    attempts.append(try_strategy_extract_state_dict(ckpt, args.output_dir))
    attempts.append(try_strategy_wrap_with_meta(sd, args.output_dir))
    attempts.append(try_strategy_onnx_passthrough(args.cache_dir, args.output_dir))

    for a in attempts:
        if a.saved_path:
            info(f"saved candidate: {a.saved_path}")

    heading("Step 4 : test each candidate with Ultralytics")
    for a in attempts:
        if a.saved_path is None:
            continue
        print(f"  testing {a.name} -> {a.saved_path.name}")
        test_with_ultralytics(a, args.test_image)

    # Anything that loaded AND ran inference: produce an annotated image.
    heading("Step 4b : annotate test image with any working candidate")
    annotated_any = False
    for a in attempts:
        if a.saved_path is None or a.load_error or a.inference_error:
            continue
        try:
            from ultralytics import YOLO
            import cv2
            model = YOLO(str(a.saved_path))
            res = model(str(args.test_image), verbose=False)
            out_path = args.output_dir / f"{a.saved_path.stem}_annotated.png"
            plot = res[0].plot() if res else None
            if plot is not None:
                cv2.imwrite(str(out_path), plot)
                info(f"  {a.name} -> saved {out_path}")
                annotated_any = True
                a.notes.append(f"annotated -> {out_path}")
        except Exception as exc:  # noqa: BLE001
            a.notes.append(f"annotation failed: {exc}")
    if not annotated_any:
        warn("no Ultralytics candidate could annotate the test image.")

    # Last-resort: prove the cached weights DO work, just not via Ultralytics.
    heading("Step 5 : native RF-DETR inference on the test image")
    attempts.append(try_strategy_native_rfdetr(args, args.test_image, args.output_dir))
    native = attempts[-1]
    if native.saved_path:
        info(f"native run produced: {native.saved_path}")
    elif native.save_error:
        warn(f"native run failed: {native.save_error}")

    return print_verdict(attempts, inspection["arch"])


if __name__ == "__main__":
    sys.exit(main())
