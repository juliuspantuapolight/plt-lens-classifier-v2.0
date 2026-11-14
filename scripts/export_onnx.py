"""Export trained MobileNet slot models (model.pt) to model.onnx next to them, and check the outputs match."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mobile_classifier import load_mobile
from src.onnx_classifier import OnnxClassifier


def export(model_dir: Path) -> float:
    processor, classifier = load_mobile(model_dir)
    network = classifier.network.eval()
    size = processor.image_size
    torch.onnx.export(network, (torch.zeros(1, 3, size, size),), str(model_dir / "model.onnx"),
                      input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    # Same random crops through both paths, including the preprocessing each one uses.
    crops = [np.random.default_rng(seed).integers(0, 256, (60, 58, 3), dtype=np.uint8) for seed in range(8)]
    with torch.inference_mode():
        expected = torch.softmax(network(processor(crops)["pixel_values"]), dim=-1).numpy()
    actual = OnnxClassifier(model_dir).probabilities(crops)
    return float(np.abs(expected - actual).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=Path, help="Folders containing slot_model.json and model.pt")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Largest allowed probability difference")
    args = parser.parse_args()
    failed = False
    for model_dir in args.model_dirs:
        difference = export(model_dir)
        print(f"{model_dir / 'model.onnx'}: max probability difference vs PyTorch {difference:.2e}")
        failed |= difference > args.tolerance
    if failed:
        sys.exit(f"ONNX output differs from PyTorch by more than {args.tolerance}; do not deploy these files.")


if __name__ == "__main__":
    main()
