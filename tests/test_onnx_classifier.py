import json

import numpy as np
import torch

from scripts.export_onnx import export
from src.mobile_classifier import build_model
from src.slot_inference import SlotInspector


def test_onnx_export_matches_pytorch_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    torch.save(build_model().state_dict(), tmp_path / "model.pt")
    (tmp_path / "slot_model.json").write_text(json.dumps({"architecture": "mobilenet_v3_small", "image_size": 96}))
    crops = [np.random.default_rng(seed).integers(0, 256, (58, 61, 3), dtype=np.uint8) for seed in range(5)]
    expected = SlotInspector(tmp_path, batch_size=2).classify(crops)  # no model.onnx yet: PyTorch path

    assert export(tmp_path) < 1e-4
    onnx = SlotInspector(tmp_path, batch_size=2, cpu_threads=2)
    assert onnx.onnx is not None
    actual = onnx.classify(crops)
    assert [label for label, _ in actual] == [label for label, _ in expected]
    assert np.allclose([c for _, c in actual], [c for _, c in expected], atol=1e-4)
    assert onnx.classify([]) == []
