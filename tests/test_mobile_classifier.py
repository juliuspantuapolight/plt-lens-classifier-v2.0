import json

import numpy as np
import pytest
import torch

from src.mobile_classifier import MobileProcessor, build_model
from src.slot_inference import SlotInspector


def test_mobile_processor_matches_training_and_inference_inputs():
    image = np.full((40, 60, 3), [255, 0, 128], dtype=np.uint8)
    processor = MobileProcessor(96)
    batch = processor([image, image])['pixel_values']
    assert batch.shape == (2, 3, 96, 96)
    assert torch.allclose(batch[0, :, 0, 0], torch.tensor([
        (1 - .485) / .229, (0 - .456) / .224, (128 / 255 - .406) / .225,
    ]))


def test_mobile_checkpoint_loads_and_classifies_partial_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.save(build_model().state_dict(), tmp_path / 'model.pt')
    (tmp_path / 'slot_model.json').write_text(json.dumps({
        'architecture': 'mobilenet_v3_small', 'image_size': 96,
    }))
    inspector = SlotInspector(tmp_path, batch_size=2, cpu_threads=2)
    crops = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(3)]
    predictions = inspector.classify(crops)
    assert len(predictions) == 3
    assert all(label in {'TLens', 'Empty'} and .5 <= confidence <= 1 for label, confidence in predictions)
    assert inspector.classify([]) == []
    with pytest.raises(ValueError, match='DINOv2 only'):
        SlotInspector(tmp_path, cpu_int8=True)
