# ML Steering Weights

The large ML artifacts for this branch are hosted on Hugging Face instead of
GitHub:

```text
HammadNaseer/qcar2-ml-steering-weights
```

## Files

| File | Purpose |
|---|---|
| `car_track_v3_lane.onnx` | Primary RF-DETR ONNX segmentation model. |
| `car_track_v3_lane.classes.txt` | Class names in model output order. |
| `fallback/best.pt` | Fallback YOLO/DL segmentation weight. |
| `fallback/resnet18_road_following.pth` | Fallback ResNet18 road-following weight. |
| `eval/onnx_runtime_smoke_test.json` | ONNX Runtime smoke-test output and top scores. |
| `eval/rfdetr_sample_predictions.json` | Sample project prediction. |

## Class Names

| Class ID | Name |
|---:|---|
| 0 | `background_class83422` |
| 1 | `lane2` |
| 2 | `traffic_light` |

## Sample Scores

These are smoke-test/sample scores, not a full validation benchmark.

| Source | Class | Score |
|---|---|---:|
| RF-DETR project sample | `lane2` | `0.9720` |
| ONNX Runtime smoke test | `lane2` | `0.8555` |

## Download

```bash
scripts/download_ml_steering_weights.sh
```

The script downloads files into the paths expected by the launch scripts:

```text
weights/car_track_v3_lane.onnx
weights/car_track_v3_lane.classes.txt
src/qcar2_autonomous_lanes/qcar2_autonomy/weights/best.pt
src/qcar2_autonomous_lanes/qcar2_autonomy/weights/resnet18_road_following.pth
```

## Primary Runtime

- Architecture: RF-DETR segmentation medium
- Runtime: ONNX Runtime
- Input: `input`, float32 NCHW `[1, 3, 432, 432]`
- Outputs:
  - `dets`: `[1, 200, 4]`
  - `labels`: `[1, 200, 3]`
  - `4647`: `[1, 200, 108, 108]`
