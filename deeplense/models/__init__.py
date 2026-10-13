import torch.nn as nn

from .cnn import LensingCNN
from .deit import DeiTTinyClassifier
from .efficientnet import EfficientNetClassifier
from .pinn import LensingPINN, PINNLoss
from .pinn_lens import LensingPINNv2, PINNv2Loss

MODELS = {
    "cnn": LensingCNN,
    "efficientnet": EfficientNetClassifier,
    "pinn": LensingPINN,
    "pinn2": LensingPINNv2,
    "deit": DeiTTinyClassifier,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["MODELS", "build_model", "LensingCNN", "EfficientNetClassifier",
           "LensingPINN", "PINNLoss", "LensingPINNv2", "PINNv2Loss", "DeiTTinyClassifier"]
