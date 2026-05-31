# car-track-nkz9u/3 Packaged ONNX

This folder contains a standalone offline ONNX package for the Roboflow model
`car-track-nkz9u/3`.

## Artifact

- Model file: `weights/car_track_nkz9u_3_rfdetr_seg_medium.onnx`
- Architecture: RF-DETR segmentation medium
- Task: instance segmentation
- Input: `input`, float32 NCHW `[1, 3, 432, 432]`
- Outputs:
  - `dets`: `[1, 200, 4]` normalized `cx, cy, w, h`
  - `labels`: `[1, 200, 3]` class logits
  - `4647`: `[1, 200, 108, 108]` mask logits
- Classes: `background_class83422`, `lane2`, `traffic_light`
- Packaged ONNX SHA256:
  `dfbac7ce35c5befc3397f722df9380ca9a9b46df1e9ea2dd6a954405cbbe1a9c`

The original Roboflow cached ONNX blob was:

`weights/roboflow_cache/shared-blobs/957888b8797d0349bf78b6ea0f124b8f`

Its SHA256 was:

`c981e0652eb1c268f1fa28f7cc4d51e8df8b9064d78c7a3e3b84dfda0d762fbd`

The packaged file differs only because metadata was embedded into the ONNX.

## Install

```bash
python3 -m pip install --upgrade onnx onnxruntime opencv-python numpy
```

For GPU runtime, install the provider matching your machine instead of plain
`onnxruntime`, then pass the provider:

```bash
python3 scripts/run_car_track_onnx.py \
  --image "lane debug_screenshot_24.05.2026.png" \
  --providers CUDAExecutionProvider,CPUExecutionProvider
```

## Verify

```bash
cd ~/rosbot_ws
python3 - <<'PY'
import onnx, onnxruntime as ort
p = "weights/car_track_nkz9u_3_rfdetr_seg_medium.onnx"
onnx.checker.check_model(onnx.load(p))
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape, i.type) for i in s.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in s.get_outputs()])
PY
```

Expected input/output summary:

```text
inputs: [('input', [1, 3, 432, 432], 'tensor(float)')]
outputs: [('dets', [1, 200, 4], 'tensor(float)'), ('labels', [1, 200, 3], 'tensor(float)'), ('4647', [1, 200, 108, 108], 'tensor(float)')]
```

## Run A Single Image

```bash
cd ~/rosbot_ws
python3 scripts/run_car_track_onnx.py \
  --image "lane debug_screenshot_24.05.2026.png" \
  --onnx weights/car_track_nkz9u_3_rfdetr_seg_medium.onnx \
  --output-dir weights/onnx_runs \
  --check-model
```

Outputs:

- Annotated image: `weights/onnx_runs/<image_stem>_onnx.png`
- JSON report: `weights/onnx_runs/<image_stem>_onnx.json`

## Run A Folder

```bash
cd ~/rosbot_ws
python3 scripts/run_car_track_onnx.py \
  --input-dir frames \
  --onnx weights/car_track_nkz9u_3_rfdetr_seg_medium.onnx \
  --output-dir weights/onnx_runs
```

## Important Notes

Roboflow returned `403 Forbidden` for direct `.pt` download with this model/key,
so a clean Roboflow `best.pt` export is not available from that account.

The PyTorch cache blob is RF-DETR, not Ultralytics YOLO. It cannot be converted
into a real Ultralytics `best.pt` by wrapping or renaming tensors. The working
portable artifact is this RF-DETR ONNX file, which runs offline through
ONNXRuntime.
