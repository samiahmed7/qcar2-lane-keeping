# Extracting a Roboflow-hosted model as a portable ONNX file and running it locally

A focused, copy-pasteable recipe. **No Roboflow SDK at inference time**, no
Docker, no Roboflow Inference Server, no internet during inference. The
output is *one* `.onnx` file (plus a tiny class-names text file) that runs
with `onnxruntime` from any Python project, any language with ONNX bindings,
on CPU or GPU.

This is the path we used to ship a learned QCar 2 lane keeper at 13 FPS on
GPU. See [RFDETR_GPU_LANE_KEEPING_JOURNEY.md](RFDETR_GPU_LANE_KEEPING_JOURNEY.md)
for the full story; this doc is the short, repeatable how-to.

> Confirmed limitation, May 2026: Roboflow **blocks direct `.pt` download**
> from the API for hosted projects (`403 Not authorized to download this
> model in pt format`). The trick below sidesteps that by using their own
> caching mechanism to materialize the weights locally, then lifting the
> ONNX out of the cache.

---

## What you get

| File | Size | Purpose |
|---|---|---|
| `weights/<your_model>.onnx` | typically 50–250 MB | the complete trained model graph + weights |
| `weights/<your_model>.classes.txt` | ~50 B | class names, one per line |
| (optional) `scripts/<your_model>_inference.py` | ~200 lines | standalone CLI: image-in → annotated PNG + JSON out |

Inference dependencies: `onnxruntime + opencv-python + numpy`. That's it.

---

## Step 0. Prerequisites

```bash
# Roboflow's `inference` package only for the ONE TIME cache fetch:
pip install inference inference-cli onnxruntime opencv-python numpy

# Optional: GPU inference
pip install onnxruntime-gpu

# For GPU on CUDA-12-targeted onnxruntime-gpu (most current builds):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 \
            nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
            nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12
```

You'll need:

1. A Roboflow account with a trained model project (e.g. `your-workspace/your-project/3`).
2. Your Roboflow API key (Settings → API Key). Treat it like a password.
3. Any one frame from your scene (a JPG / PNG) for the first cache-trigger call.

---

## Step 1. Trigger the local cache (one-time)

```bash
export ROBOFLOW_API_KEY="<your_key>"

# Pick a persistent cache location. /tmp gets wiped on reboot; do NOT use it.
export MODEL_CACHE_DIR="$PWD/weights/rf_cache"
export INFERENCE_HOME="$PWD/weights/rf_cache"   # ← the variable that holds the BIG files
mkdir -p "$INFERENCE_HOME"

python3 - <<'PY'
import os
from inference import get_model
model = get_model(
    model_id="your-workspace/your-project/3",   # ← change me
    api_key=os.environ["ROBOFLOW_API_KEY"],
)
print("downloaded + cached. task =", model.task, "names =", model.model.class_names)
PY
```

What just happened:

- `inference.get_model()` hits Roboflow once over HTTPS to fetch the model's
  serialized artifacts.
- Roboflow's cache writes a content-addressable blob store:

```
$INFERENCE_HOME/
├── usage.db                                          (tiny)
├── shared-blobs/                                     ← actual model data
│   ├── <hash_for_weights_pth>                        ← PyTorch weights, 50-250 MB
│   ├── <hash_for_weights_onnx>                       ← ONNX weights, similar size
│   ├── <hash_for_class_names.txt>                    (~50 B text)
│   ├── <hash_for_inference_config.json>              (KB)
│   └── <hash_for_model_type.json>                    (KB)
├── models-cache/
│   └── <project>-<version>-<short>/                  ← directory layout per project
│       ├── <pytorch_variant>/{model_config.json, symlinks → shared-blobs}
│       └── <onnx_variant>/   {model_config.json, symlinks → shared-blobs}
└── <project_slug>/<version>/model_type.json
```

The symlinks in `models-cache/*/<variant>/` point into `shared-blobs/`.
This is Roboflow's deduplication trick, not documented externally.

---

## Step 2. Identify which blob is the ONNX

The shared blobs are content-hashed, so you can't tell from the filename
which is which. Identify by file magic:

```bash
cd "$INFERENCE_HOME/shared-blobs"
for f in *; do head -c 32 "$f" | xxd | head -1 | sed "s|^|$f: |"; done
```

Output looks like:

```text
1364ed8...: 7b22 696d 6167 65...   {"image_pre_proc...   ← inference_config.json
29f20a6...: 7b22 6d6f 6465 6c...   {"model_type":"...    ← model_type.json
8d136b1...: 7b22 696d 6167 65...   {"image_pre_proc...   ← (another JSON variant)
9578888...: 0808 1207 7079 74...   ....pytorch...        ← ONNX  (protobuf header)
cfc5d29...: 504b 0304 0000 08...   PK..............      ← weights.pth (torch.save zip)
f90a9bc...: 6261 636b 6772 6f...   background_class...   ← class_names.txt
```

Rules:

| First bytes | Format |
|---|---|
| `PK\x03\x04` | PyTorch `torch.save` checkpoint (a zip archive) — what `.pt` files actually are |
| `\x08...` followed by `pytorch` / `onnx` ASCII somewhere in the first 1 KB | ONNX protobuf, exported from PyTorch |
| `{"...` | JSON config |
| Plain ASCII | class_names.txt |

Match the blob hashes against `models-cache/*/<onnx_variant>/weights.onnx`
(it'll be a symlink targeting the right one) or just `file` it:

```bash
file "$INFERENCE_HOME/shared-blobs/"*
# 9578888...: data
# cfc5d29...: Zip archive data, at least v0.0 to extract
# ...
```

---

## Step 3. Lift the ONNX into a clean, named file

```bash
ONNX_BLOB="$INFERENCE_HOME/shared-blobs/9578888..."   # use the hash from Step 2
CLASSES_BLOB="$INFERENCE_HOME/shared-blobs/f90a9bc..."

cp "$ONNX_BLOB"    weights/my_model.onnx
cp "$CLASSES_BLOB" weights/my_model.classes.txt
ls -lh weights/my_model.*
```

Verify ONNX is valid + inspect its IO contract:

```python
import onnxruntime as ort
sess = ort.InferenceSession("weights/my_model.onnx",
                            providers=["CPUExecutionProvider"])
for i in sess.get_inputs():
    print("input :", i.name, i.shape, i.type)
for o in sess.get_outputs():
    print("output:", o.name, o.shape, o.type)
```

You'll see something like:

```
input : input [1, 3, 432, 432] tensor(float)
output: dets   [1, 200, 4]     tensor(float)
output: labels [1, 200, K]     tensor(float)
output: <hash> [1, 200, M, M]  tensor(float)   # mask logits (only if task=segment)
```

**Write the input H×W down.** Roboflow trains at non-standard sizes
(common: 416×416, 432×432, 640×640, 800×800). The ONNX bakes this in;
you must feed exactly that size.

---

## Step 4. Pre/postprocessing — the part Roboflow's SDK normally hides

For RF-DETR-Seg models, this is the contract (verified by running side-by-side
against Roboflow's hosted inference and matching confidence to 3 decimals):

**Preprocess:** BGR camera frame
```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (W, H))                            # exactly the ONNX input size
arr = resized.astype(np.float32) / 255.0                     # [0, 1]
mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]    # ImageNet
arr = (arr - mean) / std
chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)  # 1, C, H, W
```

**Postprocess (the gotcha):** the `labels` output contains **raw logits**,
not probabilities. Roboflow's hosted API does the sigmoid for you; the bare
ONNX does not.

```python
probs = 1.0 / (1.0 + np.exp(-labels.astype(np.float32)))     # sigmoid per class
# Mask out the synthetic background class so it never wins argmax:
probs[:, 0] = -np.inf                                         # if classes[0] == "background_*"
best_cls  = probs.argmax(axis=1)
best_conf = probs[np.arange(len(probs)), best_cls]
keep      = best_conf >= confidence_threshold
```

**Boxes** are normalized `(cx, cy, w, h)` ∈ [0, 1] referenced to the ONNX
input size. To plot in the original image, scale by original (H, W).

**Masks** (segmentation models only) are `(N, M, M)` mask **logits**.
Sigmoid → upscale to the original frame → threshold at 0.5.

A working reference is at
[`scripts/rfdetr_onnx_inference.py`](../scripts/rfdetr_onnx_inference.py)
in this repo. It's <250 lines and has no Roboflow dependency.

---

## Step 5. Run it offline

```bash
python3 scripts/rfdetr_onnx_inference.py \
    --onnx    weights/my_model.onnx \
    --classes weights/my_model.classes.txt \
    --image   path/to/frame.jpg
```

Output:

- `rfdetr_outputs/<name>_annotated.png` — boxes + polygon overlay + label
- `rfdetr_outputs/<name>_predictions.json` — `[{class, confidence, bbox_xyxy}, ...]`

For a folder:

```bash
python3 scripts/rfdetr_onnx_inference.py \
    --onnx weights/my_model.onnx \
    --input-dir /path/to/frames \
    --output-dir runs/
```

---

## Step 6. Switch to GPU (12× speedup or better)

CPU is enough for batch labelling; for control loops you need GPU.

```bash
# Make CUDA 12 runtime libs visible to onnxruntime-gpu.
# These were installed in Step 0 under /usr/local/lib/python3.12/dist-packages/nvidia/*/lib
NV_LIBS=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 3 -type d -name lib | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"

python3 scripts/rfdetr_onnx_inference.py \
    --onnx    weights/my_model.onnx \
    --classes weights/my_model.classes.txt \
    --image   frame.jpg \
    --provider CUDAExecutionProvider
```

Verify it actually went to GPU:

```python
session = ort.InferenceSession("weights/my_model.onnx",
                               providers=["CUDAExecutionProvider",
                                          "CPUExecutionProvider"])
print(session.get_providers())   # should show CUDAExecutionProvider FIRST
```

Benchmark (RF-DETR-Seg-Medium, 432×432, batch 1, 10 runs):

| Provider | Latency | FPS |
|---|---|---|
| CPU (any modern x86) | ~900 ms | ~1.1 |
| GPU (RTX 3060, T4) | ~75 ms | ~13 |
| GPU + TensorRT (FP16) | typically ~10–25 ms | 40+ |

---

## Step 7. Wire to your application

This recipe is format-agnostic. Once the ONNX runs locally:

- **Python**: just import `onnxruntime`. Done.
- **ROS 2 node**: load the ONNX in `__init__`, run inference in a worker
  thread, publish results from a fast timer. Reference implementation:
  [`rfdetr_onnx_lane_node.py`](../src/qcar2_autonomous_lanes/qcar2_autonomy/qcar2_autonomy/rfdetr_onnx_lane_node.py).
- **C++**: ONNX Runtime ships an official C++ API.
  [olibartfast/rf-detr-cpp-inference](https://github.com/olibartfast/rf-detr-cpp-inference) is a working reference for RF-DETR specifically.
- **Embedded / Jetson**: export ONNX → TensorRT engine with `trtexec`.
  Same model file, just an extra `.trt` conversion step on the target device.
- **Web**: ONNX Runtime Web (`onnxruntime-web` on npm) runs ONNX in the
  browser via WebGPU/WebGL.

---

## Troubleshooting

### `403 Not authorized to download this model in pt format`
Expected. You don't need the `.pt`. Use `inference.get_model()` as in Step 1.

### `/tmp/cache` keeps re-downloading after reboot
You forgot `INFERENCE_HOME`. Setting only `MODEL_CACHE_DIR` persists
metadata but **not** the actual weights, which go through the
`inference-models` autoloader. Set BOTH:
```bash
export MODEL_CACHE_DIR=/persistent/path
export INFERENCE_HOME=/persistent/path
```

### `Got invalid dimensions for input ... Got: 448 Expected: 432`
You passed the wrong input size. The ONNX has a fixed input H×W (Step 3).
Resize to **exactly** that size before inference. Don't use any framework
helper (Ultralytics, Roboflow) that auto-pads — they round to multiples
of 32 and break the fixed-size assumption.

### `Failed to load library libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file`
CUDA 12 runtime libs missing or not on the linker path. Install them
(Step 0) and export `LD_LIBRARY_PATH` (Step 6).

### Confidences over 1.0 (e.g. `3.55`)
The model output is raw logits. Apply sigmoid before thresholding (Step 4).
`sigmoid(3.55) ≈ 0.972`.

### One "lane" instance covers the whole drivable area (not per-line)
That's not a bug; that's how your dataset was labeled on Roboflow. If you
want per-line instance segmentation you must re-label your dataset that
way and re-train.

### `Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*`
You probably have CUDA 13 driver but no CUDA 12 runtime. Install
`nvidia-cudnn-cu12 nvidia-cublas-cu12` and friends — they ship the
exact libs needed without touching the system driver. WSL2 specifically:
the driver lives in Windows, the runtime libs come from pip.

---

## Compare to documented approaches

| Approach | Pros | Cons |
|---|---|---|
| Roboflow Hosted API (default) | Just `model.infer(image)`. Always up-to-date. | ~1–2 s/call cloud RTT. Requires internet. Counts against API quota. |
| Roboflow `inference` package | Local Python wrapper around their cache. Handles pre/post automatically. | ~250 MB+ deps tree (CUDA libs, pydantic, fastapi, etc). Architecture-coupled to Roboflow's class hierarchy. |
| Roboflow Inference Server (Docker) | Their documented "offline" path. HTTP API to a local container. | Heavy. Needs Docker + a 1+ GB image. HTTP overhead. |
| **This recipe: bare ONNX + onnxruntime** | One model file. Minimal deps. Same accuracy as cloud. **GPU and CPU.** No Docker. Works in any language with ONNX bindings. | You write the pre/post yourself (template included). Tightly coupled to YOUR model's input shape + output contract. |

---

## Why this isn't well-documented externally

[Roboflow's offline-mode docs](https://docs.roboflow.com/deploy/enterprise-deployment/offline-mode)
assume the **Inference Server** (Docker) is your offline target. That's a
fine production choice but heavy. The Roboflow [community thread on running
without internet](https://discuss.roboflow.com/t/running-it-locally-without-using-internet/9207)
points users at the same Docker server.

What's missing publicly:

1. The `shared-blobs/` content-addressable cache layout — there's no public
   doc explaining how to read it directly.
2. The `INFERENCE_HOME` env var — never documented in the same place as
   `MODEL_CACHE_DIR`, and the two have non-obvious responsibilities.
3. The "you can just `cp` the ONNX out and use it" trick — works today,
   could be made unsupported by Roboflow in any future SDK version.
4. The RF-DETR-specific gotcha that the ONNX outputs raw logits, requiring
   external sigmoid.
5. The CUDA 12 runtime-lib pin needed for `onnxruntime-gpu` on a CUDA 13
   driver host.

This document fills those five gaps. If you find newer Roboflow output that
shifts any of these, please update the corresponding step. Verified working
against `inference == 1.2.10`, `onnxruntime-gpu == 1.26.0`, Roboflow's
`car-track-nkz9u/3` (RF-DETR-Seg-Medium), May 2026.

---

## Sources

- [Roboflow Offline Mode docs](https://docs.roboflow.com/deploy/enterprise-deployment/offline-mode)
- [Roboflow blog — Deploy Computer Vision Models Offline](https://blog.roboflow.com/deploy-computer-vision-models-offline/)
- [Roboflow Inference docs — Models](https://inference.roboflow.com/models/)
- [roboflow/inference repo](https://github.com/roboflow/inference)
- [DeepWiki — Roboflow Inference Model Ecosystem](https://deepwiki.com/roboflow/inference/5-sdk-and-cli)
- [Roboflow community — Running it locally without internet](https://discuss.roboflow.com/t/running-it-locally-without-using-internet/9207)
- [olibartfast/rf-detr-cpp-inference (GitHub)](https://github.com/olibartfast/rf-detr-cpp-inference)
- [RF-DETR Segmentation guide (LearnOpenCV)](https://learnopencv.com/rf-detr-segmentation-real-time-detection-instance-segmentation-guide/)
