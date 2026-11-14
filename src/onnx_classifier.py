"""ONNX Runtime slot classifier: the exported MobileNet with the same preprocessing, without torch."""
from pathlib import Path
import json

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxClassifier:
    def __init__(self, model_dir: Path, cpu_threads: int | None = None):
        config = json.loads((model_dir / "slot_model.json").read_text())
        self.image_size = config["image_size"]
        self.labels = config.get("labels", ["Empty", "TLens"])
        self.cpu_threads = cpu_threads
        options = ort.SessionOptions()
        if cpu_threads:
            options.intra_op_num_threads = cpu_threads
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"])

    def preprocess(self, rgb_images: list[np.ndarray]) -> np.ndarray:
        # Matches MobileProcessor: scale to [0, 1], bilinear resize, ImageNet normalisation, NCHW.
        # shortcut: no antialiasing, which only matters if crops are larger than image_size (they are ~60-80 px); use INTER_AREA then.
        size = (self.image_size, self.image_size)
        batch = np.stack([cv2.resize(image.astype(np.float32) / 255, size, interpolation=cv2.INTER_LINEAR)
                          for image in rgb_images])
        return np.ascontiguousarray(((batch - MEAN) / STD).transpose(0, 3, 1, 2), dtype=np.float32)

    def probabilities(self, rgb_images: list[np.ndarray]) -> np.ndarray:
        logits = self.session.run(None, {"pixel_values": self.preprocess(rgb_images)})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
