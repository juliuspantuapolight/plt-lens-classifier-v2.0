import json

import cv2
import numpy as np
import pytest

from src.corrections import apply_corrections, parse_correction
from src.slot_inference import SlotInspector
from tests.test_slot_inference import tiny_classifier  # noqa: F401  (pytest fixture)


@pytest.fixture
def saved_result(tiny_classifier, tmp_path):  # noqa: F811
    image = tmp_path / "tray.png"
    cv2.imwrite(str(image), np.zeros((800, 800, 3), dtype=np.uint8))
    result, rendered = SlotInspector("unused").inspect(image, "packaged", True, 100)
    result["annotated_image"] = str(tmp_path / "tray_inspection.jpg")
    cv2.imwrite(result["annotated_image"], rendered)
    json_path = tmp_path / "tray_inspection.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")
    return json_path


def slot(result, row, column):
    return next(s for s in result["slots"] if (s["row"], s["column"]) == (row, column))


def test_correction_updates_counts_keeps_model_answer_and_redraws(saved_result):
    before = json.loads(saved_result.read_text())
    image_before = cv2.imread(before["annotated_image"])

    result = apply_corrections(saved_result, {(5, 5): "TLens"})

    assert json.loads(saved_result.read_text()) == result
    assert (slot(result, 5, 5)["prediction"], slot(result, 5, 5)["model_prediction"]) == ("TLens", "Empty")
    assert (result["occupied_count"], result["empty_count"]) == (before["occupied_count"] + 1, before["empty_count"] - 1)
    assert result["occupancy_matrix"][4][4] == "TLens"
    assert result["corrected_count"] == 1
    assert result["reviewed_at"]
    x, y = round(slot(result, 5, 5)["center_x"]), round(slot(result, 5, 5)["center_y"])
    patch = cv2.imread(result["annotated_image"])[y - 12:y + 13, x - 12:x + 13]
    assert (patch[..., 1] > 200).any() and not np.array_equal(patch, image_before[y - 12:y + 13, x - 12:x + 13])


def test_reverting_a_correction_drops_model_prediction(saved_result):
    apply_corrections(saved_result, {(5, 5): "TLens"})
    result = apply_corrections(saved_result, {(5, 5): "Empty"})

    assert "model_prediction" not in slot(result, 5, 5)
    assert result["corrected_count"] == 0


def test_sticker_positions_cannot_be_corrected(saved_result):
    with pytest.raises(ValueError, match="not a classified position"):
        apply_corrections(saved_result, {(1, 1): "TLens"})


@pytest.mark.parametrize("text", ["3,4=Uncertain", "3=TLens", "a,b=Empty"])
def test_parse_correction_rejects_bad_input(text):
    with pytest.raises(Exception):
        parse_correction(text)
