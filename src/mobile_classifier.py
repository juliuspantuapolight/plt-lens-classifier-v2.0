"""MobileNetV3 slot model and shared RGB preprocessing."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import functional as F
from transformers import BatchFeature

LABELS = {0: "Empty", 1: "TLens"}


def build_model(pretrained: bool = False):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    return model


class MobileProcessor:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, images, return_tensors="pt"):
        tensors = []
        for image in images:
            tensor = F.to_tensor(np.array(image, copy=True, order="C"))
            tensor = F.resize(tensor, [self.image_size, self.image_size], antialias=True)
            tensors.append(F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        return BatchFeature({"pixel_values": torch.stack(tensors)})


class MobileClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.network = model
        self.config = SimpleNamespace(id2label=LABELS)

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.network(pixel_values))


def load_mobile(model_dir: Path):
    config = json.loads((model_dir / "slot_model.json").read_text())
    if config["architecture"] != "mobilenet_v3_small":
        raise ValueError("Unsupported slot model architecture")
    model = build_model()
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    return MobileProcessor(config["image_size"]), MobileClassifier(model)
