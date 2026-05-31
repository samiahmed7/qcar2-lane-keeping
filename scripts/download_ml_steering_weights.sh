#!/usr/bin/env bash
# Download ML steering weights from Hugging Face into the paths expected by
# the QCar2 ROS 2 autonomy scripts.
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ID="${HF_REPO_ID:-HammadNaseer/qcar2-ml-steering-weights}"

if ! command -v hf >/dev/null 2>&1; then
    echo "Missing Hugging Face CLI: install with 'pipx install huggingface_hub' or see https://hf.co/cli"
    exit 1
fi

mkdir -p weights src/qcar2_autonomous_lanes/qcar2_autonomy/weights

hf download "${REPO_ID}" \
    car_track_v3_lane.onnx \
    car_track_v3_lane.classes.txt \
    --local-dir weights

hf download "${REPO_ID}" \
    fallback/best.pt \
    fallback/resnet18_road_following.pth \
    --local-dir /tmp/qcar2_ml_steering_weights

cp /tmp/qcar2_ml_steering_weights/fallback/best.pt \
    src/qcar2_autonomous_lanes/qcar2_autonomy/weights/best.pt
cp /tmp/qcar2_ml_steering_weights/fallback/resnet18_road_following.pth \
    src/qcar2_autonomous_lanes/qcar2_autonomy/weights/resnet18_road_following.pth

echo "Downloaded ML steering weights from ${REPO_ID}"
