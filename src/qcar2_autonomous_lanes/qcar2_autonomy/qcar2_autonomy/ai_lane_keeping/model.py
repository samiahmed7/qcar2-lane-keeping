"""PyTorch model loading for HSHL/JetRacer road-following weights."""
from pathlib import Path


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required for ai_lane_keeper_node. Install torch and '
            'torchvision in the ROS environment before launching AI inference.'
        ) from exc
    return torch


def require_torchvision_models():
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            'torchvision is required for ResNet model construction.'
        ) from exc
    return models


def select_device(requested):
    """Return a torch.device from 'auto', 'cpu', 'cuda', etc."""
    torch = require_torch()
    name = str(requested).strip().lower()
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def build_resnet(architecture='resnet18', output_dim=2):
    """Construct the JetRacer regression network.

    NVIDIA JetRacer road-following trains a ResNet backbone with its final fully
    connected layer replaced by a 2-output regression head. The two outputs are
    an image-space direction vector (x, y); steering is computed with atan2(x,y).
    Some custom exports use one output directly for steering, so output_dim is
    parameterized.
    """
    torch = require_torch()
    models = require_torchvision_models()
    arch = str(architecture).strip().lower()

    if arch == 'resnet18':
        model = models.resnet18(weights=None)
    elif arch == 'resnet34':
        model = models.resnet34(weights=None)
    else:
        raise ValueError(f'unsupported architecture: {architecture}')

    model.fc = torch.nn.Linear(model.fc.in_features, int(output_dim))
    return model


def load_resnet_weights(model_path, architecture='resnet18', output_dim=2, device='auto'):
    """Load a pure PyTorch state_dict .pth file and return an eval() model."""
    torch = require_torch()
    device_obj = select_device(device)
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'model_path does not exist: {path}')

    model = build_resnet(architecture=architecture, output_dim=output_dim)
    checkpoint = torch.load(str(path), map_location=device_obj)

    # Public JetRacer examples usually save a raw state_dict, but this also
    # accepts a small wrapper dict if a later fine-tuning script stores metadata.
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()
    return model, device_obj

