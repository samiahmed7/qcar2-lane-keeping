"""Image preprocessing for JetRacer-style ResNet road-following models."""
import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_top(bgr, crop_top_ratio):
    """Drop sky/ceiling rows before resizing, keeping the road-dominant region."""
    ratio = min(max(float(crop_top_ratio), 0.0), 0.85)
    if ratio <= 0.0:
        return bgr
    y0 = int(round(bgr.shape[0] * ratio))
    return bgr[y0:, :]


def preprocess_bgr_for_resnet(
    bgr,
    input_width=224,
    input_height=224,
    crop_top_ratio=0.0,
):
    """Return a CHW float32 tensor array normalized like torchvision ResNet.

    The public JetRacer road-following notebooks use a ResNet backbone trained
    on RGB camera images and ImageNet normalization. ROS/cv_bridge gives us BGR,
    so the conversion order is:

        ROS Image -> OpenCV BGR -> crop -> resize 224x224 -> RGB
        -> scale to [0, 1] -> normalize -> CHW

    The returned value is a NumPy array. The ROS node converts it to a torch
    tensor only after the model is loaded, which keeps this helper testable on
    machines that do not have PyTorch installed.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError('empty image')

    road = crop_top(bgr, crop_top_ratio)
    resized = cv2.resize(
        road,
        (int(input_width), int(input_height)),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

