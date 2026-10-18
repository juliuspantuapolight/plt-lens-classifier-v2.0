from pathlib import Path

from src.inspection_io import inspection_output_paths
from src.tray_geometry import TRAY_CONFIGS

import cv2
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from transformers import BatchFeature

from src.slot_inference import SlotInspector


@pytest.fixture
def tiny_classifier(monkeypatch):
    class Processor:
        def __call__(self, images, return_tensors):
            return BatchFeature({"pixel_values": torch.tensor(
                np.asarray([image.mean(axis=(0, 1)) for image in images]), dtype=torch.float32,
            ) / 255})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(3, 2)
            self.config = SimpleNamespace(id2label={0: "Empty", 1: "TLens"})
            with torch.no_grad():
                self.classifier.weight.copy_(torch.tensor([[-2.] * 3, [2.] * 3]))
                self.classifier.bias.copy_(torch.tensor([2., -2.]))

        def forward(self, pixel_values):
            return SimpleNamespace(logits=self.classifier(pixel_values))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("src.slot_inference.AutoImageProcessor.from_pretrained", lambda *a, **kw: Processor())
    monkeypatch.setattr("src.slot_inference.AutoModelForImageClassification.from_pretrained", lambda *a, **kw: Model())


def test_int8_preserves_clear_predictions_and_handles_partial_batch(tiny_classifier):
    crops = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (0, 255, 0)]
    fp32 = SlotInspector("unused", batch_size=2)
    int8 = SlotInspector("unused", batch_size=2, cpu_int8=True)

    assert [label for label, _ in fp32.classify(crops)] == ["Empty", "TLens", "Empty"]
    assert [label for label, _ in int8.classify(crops)] == ["Empty", "TLens", "Empty"]
    assert fp32.model.classifier.weight.dtype == torch.float32
    assert int8.model.classifier.weight().dtype == torch.qint8
    assert int8.classify([]) == []


def test_inspection_reports_runtime_and_stage_timings(tiny_classifier, tmp_path):
    image = tmp_path / "tray.png"
    cv2.imwrite(str(image), np.zeros((160, 160, 3), dtype=np.uint8))
    inspector = SlotInspector("unused", batch_size=32)

    result, rendered = inspector.inspect(image, "packaged", True, 100)

    assert result["total_positions"] == result["empty_count"] == 255
    assert result["runtime"]["precision"] == "float32"
    assert result["runtime"]["device"] == "cpu"
    assert set(result["timings_seconds"]) == {
        "image_read", "alignment_and_crops", "recenter_apertures", "classification", "quality_and_rendering",
    }
    assert all(seconds >= 0 for seconds in result["timings_seconds"].values())
    assert rendered.shape == (160, 160, 3)


def test_packaged_qr_slots_are_fixed_empty_positions():
    assert len(TRAY_CONFIGS["packaged"]["always_empty"]) == 6


def test_each_image_gets_matching_image_and_json_output_names():
    output_image, output_json = inspection_output_paths(Path("outputs"), Path("tray-001.jpg"))

    assert output_image == Path("outputs/tray-001_inspection.jpg")
    assert output_json == Path("outputs/tray-001_inspection.json")
