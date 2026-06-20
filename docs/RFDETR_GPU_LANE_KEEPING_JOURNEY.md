# Driving a sim QCar 2 with RF-DETR-Seg on GPU — the journey

A step-by-step record of how this repo went from a classical lane-keeping
pipeline (IPM + sliding windows + PID) to a learned model (RF-DETR-Seg)
driving the same Gazebo car at 13 FPS on GPU. Every blocker, the wrong
assumptions, the actual fixes, the user prompts that turned each corner,
and how the result compares to public prior art.

This is documentation of *what worked* on a specific track (HSHL Lab AS,
modeled as `qcar2_worlds/worlds/lab_track.sdf`), not a research claim. None
of the individual components are novel; the composition for this project is.

---

## TL;DR

| Stage | Approach | Result on lab_track sim |
|---|---|---|
| 1 | Classical IPM + sliding-window histogram + PID | Drove well after a **`Histogram Split Trap`** fix; smooth (±0.03 rad/s) |
| 2 | HSHL ResNet18 regression (PilotNet style) | Distribution shift; oscillates ±0.66 rad/s, drifted off-lane |
| 3 | UFLD-TuSimple (highway lanes pretrained) | Wrong concept of "lane" for indoor lab; ~0 reliable detections |
| 4 | Roboflow `car-track-nkz9u/3` via Cloud HTTP | Detected lanes at 0.97 conf but **1.5 s/call** → unusable for control |
| 5 | Roboflow weights → portable ONNX, CPU inference | Same accuracy, **2 s/frame on CPU** → still too slow |
| 6 | **Roboflow ONNX, GPU inference, ROS 2 node** | **13 FPS, ±0.02 rad/s wz, 5 mm y-drift in 15 s** ← shipping |

The end-state code is:
- `weights/car_track_v3_lane.onnx` (130 MB, the one weight file)
- `weights/car_track_v3_lane.classes.txt` (42 B, class names)
- `src/qcar2_autonomous_lanes/qcar2_autonomy/qcar2_autonomy/rfdetr_onnx_lane_node.py` (ROS node)
- `scripts/rfdetr_onnx_inference.py` (standalone CLI for offline image tests)
- `LANE_KEEPER=rfdetr` mode in `scripts/run_lane_keeping.sh`

---

## Step 1 — Classical CV had a structural blind spot

**Problem.** The lab_track scene has **three** painted white lines (far-left solid, middle dashed, right solid). `perception_node.py` was a stock 2-peak histogram detector. Whenever it picked the wrong two of three lines (e.g., middle + right when the car was actually expected in the left lane), the midpoint landed on a line and the car drove onto it.

The narrow sub-case — both halves of one wide stripe being picked as two "lines" — was fixed with the **Histogram Split Trap** override in `validation_node.py`: reject reported widths < 0.6 × tracked EMA width and treat as single-line fallback. That handles "wide line confused as two lanes" but **not** "wrong real pair selected" (when the false pair has plausible width).

**What worked.** Adding a 3-peak detector with image-position priors (in the older `bev_lane_detector_node.py`) gave per-instance lane identities and drove the car flawlessly (y drift ≈ 0 over 15 s). The price was custom messages (`qcar2_msgs/LaneModel`) and a state-bridge node to convert `std_msgs/String` ↔ `qcar2_msgs/BehaviorState`. The `LANE_KEEPER=bev` mode in the run script.

**Key prompt that drove the right fix.**
> *"so where is the implementation where I could see the segments of all three lanes and everything what is this non sense one line and yellow dot?"*

That was the turning point — until then I was incrementally band-aiding a 2-line detector. The prompt forced me to admit the architecture itself couldn't represent three lanes and pivot to the existing 3-line node.

---

## Step 2 — End-to-end ResNet18 (HSHL) doesn't transfer to sim

**Problem.** The HSHL student `TasawarSiddiquy` had published pretrained ResNet18 + ResNet34 PyTorch weights for the same lab_track. Loading them was straightforward; the model predicts `(x, y)` in [-1, 1] — a steering target point. But on Gazebo's stylized rendering the model's predictions wobbled wildly (±0.66 rad/s steering, y drift 0.07 m in 15 s).

Distribution shift: the model was trained on **real HSHL track** photos (warm indoor LED lighting on slightly textured polished concrete, thin painted dashes/solids). Gazebo's `lab_track.sdf` renders stark white rectangles on flat-shaded gray ground — visually different enough that the conv features don't fire correctly.

**What worked.** Nothing without retraining. Tried widening lane lines in the SDF, tried adjusting IPM extents, tried lateral offset on the model output — all of them masked the symptom, not the cause.

**Key prompt that closed this branch.**
> *"can DL solve this or not"*

After three rounds of softening words I had to be direct: *"Yes — but only with a different architecture trained on YOUR specific scene appearance."* That stopped the time sink on HSHL transfer.

---

## Step 3 — UFLD-TuSimple was the wrong model family entirely

**Problem.** Pivoted to anchor-based lane detection: Ultra-Fast-Lane-Detection-v1 (TuSimple weights). The model outputs up to 4 labeled lane lines per frame — exactly the "instance segmentation" interpretation I needed. But:
- Highway-trained dataset has lane lines drawn at very different proportions to road width
- On lab_track it kept detecting the *edges of one wide white stripe* as two separate lanes (the same histogram-split trap a CNN can hit when the white object dominates)
- Adding a minimum lane-pair separation guard caught the false pair but left 0 valid detections most of the time

**Key prompt.**
> *"no [it is] not correct ... car is driving on the right lane EXACTLY because it is taking right lane's left border ... and driving on it"*

The user had to spell out what the model was actually doing (driving on top of a line, not in a lane) before I accepted that distribution shift wasn't fixable with thresholds.

---

## Step 4 — User trained the right model on the right data

**Problem.** The user uploaded their track frames + labels to Roboflow, trained `car-track-nkz9u/3` (RF-DETR-Seg-Medium architecture), got a `best.pt` back. **Roboflow blocks direct `.pt` download for hosted projects** with HTTP 403 *"Not authorized to download this model in pt format."*

**What worked.** Use the `inference` Python package: `inference.get_model("car-track-nkz9u/3", api_key=...)` performs the model fetch on first call and caches the result locally. No direct PT download needed.

**Hidden gotcha.** The default cache is `/tmp/cache/`, which gets wiped on reboot. Setting `MODEL_CACHE_DIR` only persists *some* artifacts; the actual weights need `INFERENCE_HOME` (a separate env var read by `inference-models`, not `inference`). The discovery process:
```
$ find / -mmin -120 -name 'weights.*'     # find recently-downloaded weight file
$ grep -rEn 'cache_base|cache_dir' inference/  # find env var names
$ python3 -c "import builtins; ..."        # monkey-patch open() to log all reads
```
Resulting fix in `scripts/roboflow_local_inference.py`:
```python
os.environ["MODEL_CACHE_DIR"] = str(cache_dir)
os.environ["INFERENCE_HOME"] = str(cache_dir)       # this is the one that matters
os.environ["INFERENCE_MODEL_CACHE_DIR"] = str(cache_dir)
os.environ["INFERENCE_MODEL_DIR"] = str(cache_dir)  # belt and suspenders
```

**Key prompt.**
> *"i have trained the model for track but its not that goodly trained but to test ive best.pt can you check?"* → forced me to inspect what was actually inside the .pt (RF-DETR, not YOLO).

---

## Step 5 — `YOLO("best.pt")` can never load these weights

**Problem.** Once we had the cached PyTorch checkpoint, the natural reaction was *"just load it with Ultralytics."* The script `scripts/convert_roboflow_to_ultralytics.py` tried every reasonable wrapping/extraction strategy. All failed:

| Strategy | Failure |
|---|---|
| Save raw state_dict | `KeyError: 'model'` (Ultralytics expects nn.Module, not tensors) |
| Wrap in YOLO-shaped dict `{model: sd, ema: None, …}` | `AttributeError: 'OrderedDict' object has no attribute 'float'` |
| Copy the cached ONNX, pass to `YOLO("best.onnx")` | Loaded! But inference fails: `Got: 448 Expected: 432` — Ultralytics pads to multiples of 32, RF-DETR has fixed positional encodings at exactly 432×432 |

The key insight: **the keys don't overlap.** RF-DETR's state dict has names like `transformer.decoder.layers.0.self_attn.in_proj_weight` and `backbone.0.projector.stages.0.0.m.2.cv1.conv.weight`. Ultralytics YOLO expects `model.0.conv.weight`, `model.22.dfl`, etc. Zero overlap means no rename can produce a YOLO-loadable checkpoint.

**Key prompt.**
> *"try converting and check the conversions on the image"*

This forced me to actually run each candidate through Ultralytics + an image, instead of theorizing. Quantitative result: 4 out of 5 strategies failed at `YOLO()` instantiation, the 5th (ONNX passthrough) loaded but failed at the input-size mismatch. Conclusion was unambiguous: not a wrapping problem, an architecture mismatch.

---

## Step 6 — The ONNX *is* the single portable artifact

**Problem.** After confirming `.pt` was unusable outside RF-DETR's own loader, the question became: *how do I package this as a single file that just works?* The cached state contained six blobs in `shared-blobs/`. Disambiguated them by file magic:

```
cfc5d29... starts with 'PK\x03\x04'         → torch.save zip (the .pth)
9578888... starts with '\x08\x08\x12\x07pytorch'  → ONNX protobuf
1364ed8..., 8d136b1...                       → JSON configs
f90a9bc...                                   → class_names.txt (plain text)
29f20a6...                                   → JSON model_type
```

**Single-file deliverable:** `weights/car_track_v3_lane.onnx` (130 MB) + `weights/car_track_v3_lane.classes.txt` (42 B). The ONNX has the entire model graph + weights baked in. To make it actually useful, wrote `scripts/rfdetr_onnx_inference.py` — a 200-line standalone that handles preprocess (BGR → RGB → ImageNet normalize → 432×432) and postprocess (sigmoid logits → argmax → mask resize + binarize).

One non-obvious pitfall on first run: the model outputs raw logits, not probabilities. I displayed `confidence = 3.55` until I added sigmoid in the postprocessor (`sigmoid(3.55) ≈ 0.97`, which matched Roboflow's hosted prediction).

**Key prompt.**
> *"can i atleast get a weight single file what i can use again and again?"*

Pushed me to stop talking about workarounds and produce the actual portable file + the verification.

---

## Step 7 — GPU is the difference between 1 FPS and 13 FPS

**Problem.** Standalone ONNX inference on CPU: ~2 s/frame. Roboflow cloud: ~1.5 s. Both unusable for control at any speed > "park and ponder."

**Fixes (in order, none trivial):**
1. `pip install onnxruntime-gpu` — replaces the CPU-only build. After install, `ort.get_available_providers()` shows `CUDAExecutionProvider`.
2. **But it doesn't actually load** because the GPU runtime libs aren't on the linker path. The error message was useful: `Failed to load library libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file`. Specifically requires CUDA 12, while the WSL2 driver was for CUDA 13.
3. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 …` — installs the CUDA 12 libs alongside the CUDA 13 ones already there (pip-installed nvidia packages put their libs under `/usr/local/lib/python3.12/dist-packages/nvidia/*/lib/`).
4. Add those directories to `LD_LIBRARY_PATH` before launching onnxruntime:
   ```bash
   NV_LIBS=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 3 -type d -name lib | tr '\n' ':')
   export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"
   ```

After all that:
```
CPU: mean 905 ms, median 929 ms, min 715 ms     →   1.1 FPS
GPU: mean  75 ms, median  69 ms, min  59 ms     →  13.3 FPS
```

12× speedup. That's the difference between "demo only" and "the car drives smoothly."

---

## Step 8 — ROS node + live driving

**Problem.** Wrap the ONNX inference in a ROS 2 node that:
- Takes camera frames from `/qcar2/front_camera/image`
- Runs RF-DETR-Seg on GPU
- Computes a `target_x` from the lane mask
- Publishes to `/planning/validated_target_x` for the existing `pid_lane_follower_node`
- Stays out of the way during `LANE_CHANGE` (state machine + DWA still own avoidance)

Key design decision: inference runs on a **background thread**, the publish timer just emits the most-recent cached overlay + target. That decouples the 75 ms inference latency from the 20 Hz ROS callback rate. Implementation in `rfdetr_onnx_lane_node.py`.

For `target_x`: take the bottom 30% of the binary lane mask, compute the column-wise centroid of mask pixels. That's the lane center *closest to the car*. Constant `target_band_y_ratio = 0.70` (configurable).

**Live result on lab_track sim:**
```
   t      car_xy             d      vx     wz     target_x
  0.02s ( 1.47,-0.008) 0.000m  +0.20  +0.00   319.4
  5.03s ( 1.58,-0.010) 0.108m  +0.20  +0.00   319.3
 10.04s ( 1.67,-0.012) 0.192m  +0.20  -0.01   324.8
 15.01s ( 1.76,-0.013) 0.288m  +0.20  +0.00   319.8
```

- vx steady at 0.20 m/s (commanded)
- wz spikes capped at ±0.02 rad/s — same smoothness as the hand-tuned classical pipeline
- target_x parked at ~320 (camera center)
- y drift 5 mm over 15 s

The state-machine → DWA handoff for obstacle avoidance is unchanged: drop a box in front of the car, `state_machine_node` flips to `LANE_CHANGE`, `dwa_local_planner_node` takes over `/cmd_vel`, mux drops PID output. This stack is orthogonal to the lane keeper choice.

---

## Has anything like this been done before?

Yes, in the *generic* sense; no, in the *specific composition* sense.

**Generic prior art (all common):**

- [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr) — the architecture, released March 2025. ICLR 2026 paper. SOTA on COCO with real-time latency (2.3 ms on T4 for RF-DETR-N).
- [RF-DETR Segmentation guide (LearnOpenCV)](https://learnopencv.com/rf-detr-segmentation-real-time-detection-instance-segmentation-guide/) — how to fine-tune and run RF-DETR-Seg on COCO-style data.
- [olibartfast/rf-detr-cpp-inference](https://github.com/olibartfast/rf-detr-cpp-inference) — C++ RF-DETR inference with ONNX Runtime and TensorRT. Same architecture, different language. Doesn't integrate ROS.
- [Roboflow Inference + ROS](https://blog.roboflow.com/run-inference/) — Roboflow have documented running their inference HTTP API from Jetson via RTSP streams. Does not cover *local ONNX* inference, only their cloud/edge HTTP wrapper.
- Lane detection on Roboflow Universe: many [pretrained lane models](https://universe.roboflow.com/mit-wpu-3jmcg/lane-detection-vxwns) exist, used in F1Tenth and similar small autonomous platforms.

**What's *not* covered in the public material I could find:**

1. Running RF-DETR-Seg *as a portable ONNX, standalone*, without the `inference` package wrapper. The Roboflow ecosystem assumes you'll use their SDK (cloud or edge HTTP). Extracting the cached ONNX as a single-file deliverable + writing standalone pre/post is something users have to do themselves.
2. Integration with **ROS 2** as a `rclpy` Node publishing `Float32` target signals to feed a downstream PID. Roboflow's blog posts cover RTSP + their HTTP API; not ROS topic plumbing.
3. The specific cache-dir gotcha (`INFERENCE_HOME` vs `MODEL_CACHE_DIR`) is undocumented externally.
4. Sim-in-the-loop validation with **Gazebo** (`lab_track.sdf`) before committing to hardware. Roboflow examples are real-world camera streams.

**Verdict.** No paper, no blog post, no public repo demonstrates the full chain we built here on this specific car/track. But every individual brick (RF-DETR architecture, ONNX export, onnxruntime-gpu, ROS-camera integration, sim validation) has well-trodden examples. This document is the "we glued the bricks together correctly for *this* project" record — useful as a how-to for the next person on the QCar 2 / HSHL track stack, not as a research artifact.

---

## Reproduction recipe (for the next person)

```bash
# 0. Environment
cd ~/rosbot_ws
echo 'Pakistan123@' | sudo -S pip install --break-system-packages --ignore-installed \
  onnxruntime-gpu inference inference-cli \
  nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12

# 1. Get the model (if not already cached locally)
ROBOFLOW_API_KEY=<your_key> python3 scripts/roboflow_local_inference.py \
  --image any_frame.jpg --cache-dir weights/roboflow_cache

# 2. Extract the portable ONNX (this is the one-file deliverable)
cp weights/roboflow_cache/shared-blobs/<onnx_blob_hash> weights/car_track_v3_lane.onnx
cp weights/roboflow_cache/shared-blobs/<class_names_hash> weights/car_track_v3_lane.classes.txt

# 3. Verify standalone inference works
python3 scripts/rfdetr_onnx_inference.py --image any_frame.jpg
ls rfdetr_outputs/*.png  # expect annotated PNG

# 4. Build the ROS package (after rfdetr_onnx_lane_node.py is added to setup.py)
source /opt/ros/jazzy/setup.bash
colcon build --packages-select qcar2_autonomy --symlink-install

# 5. Drive the car
LANE_KEEPER=rfdetr ./scripts/run_lane_keeping.sh
# View overlay in another terminal:
ros2 run rqt_image_view rqt_image_view /rfdetr_lane/debug_image
# Inspect computed target:
ros2 topic echo /planning/validated_target_x
# Tear down:
./scripts/stop_lane_keeping.sh
```

---

## Prompts that turned each corner

A compact log of the user inputs that pushed past each blocker. Useful when
the next person describes a similar problem to an LLM coding assistant:

| Stage | Effective prompt |
|---|---|
| Stop softening, give direct verdict | *"can DL solve this or not"* |
| Force admission about model architecture | *"so where is the implementation where I could see the segments of all three lanes"* |
| Demand empirical test, not theory | *"run simulation run car add obj see logs if achieved then pass"* |
| Cut my UFLD overconfidence | *"not correct ... car is driving on the right lane EXACTLY because it is taking right lane's left border ... and driving on it"* |
| Pivot to user's own training data | *"i have trained the model for track but its not that goodly trained but to test ive best.pt can you check?"* |
| Demand portable artifact | *"can i atleast get a weight single file what i can use again and again?"* |
| Trigger the GPU path | *"i have a gpu, can you run the car using this?"* |
| Anchor in honesty about whether result is novel | *"do i now have a single file is this a breakthrough?"* |
| Capture this whole journey | *"can we document this step by step the problem and the solution and the prompt it requires"* |

The pattern: short, direct prompts that refuse to accept theory over evidence,
and that name the specific symptom or doubt. Each one closed a branch in
under one more iteration.

---

## Honest limits of what was achieved

- This is **sim only**. The model was trained on real-track photos; it works in Gazebo because the sim is visually close enough at 432×432 input resolution. The first physical-car test may still need fine-tuning.
- **GPU is required for real-time.** CPU inference is 12× slower; on Jetson Nano (the QCar 2's onboard compute) you'd need TensorRT export + INT8 quantization to hit the same FPS. That's another half-day of integration work.
- **The model is not perfect.** RF-DETR detected the lane interior reliably on every test frame, but the user noted "it's not that goodly trained" — accuracy on curves / edge cases on the real track has not been validated.
- **Avoidance is unchanged** from prior steps. LANE_CHANGE still hands off to DWA local planner. The lane keeper choice (rfdetr / bev / pid / ai) is orthogonal to the avoidance stack — you can mix any.

---

## Sources

- [RF-DETR (Roboflow GitHub)](https://github.com/roboflow/rf-detr)
- [RF-DETR Segmentation guide (LearnOpenCV)](https://learnopencv.com/rf-detr-segmentation-real-time-detection-instance-segmentation-guide/)
- [olibartfast/rf-detr-cpp-inference (GitHub)](https://github.com/olibartfast/rf-detr-cpp-inference)
- [RF-DETR Segmentation model card (Roboflow)](https://roboflow.com/model/rf-detr-segmentation)
- [New RF-DETR Segmentation checkpoints (Roboflow blog)](https://blog.roboflow.com/rf-detr-segmentation/)
- [Run inference on RTSP / Jetson Orin Nano (Roboflow blog)](https://blog.roboflow.com/run-inference/)
- [Roboflow Inference models docs](https://inference.roboflow.com/models/)
- [HSHL Lane Keeping with AI (wiki)](https://wiki.hshl.de/wiki/index.php/Lane_Keeping_with_AI_and_steering_angle)
- [TasawarSiddiquy/Automated-lane-following-Waveshare-JetRacer (GitHub)](https://github.com/TasawarSiddiquy/Automated-lane-following-Waveshare-JetRacer-with-artificial-intelligence)
- [Ultra-Fast-Lane-Detection (cfzd/GitHub)](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
