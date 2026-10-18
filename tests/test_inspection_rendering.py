from dataclasses import replace

import cv2
import numpy as np

from src.inspection_rendering import AnnotationLayout
from src.tray_geometry import align_tray, extract_slots


def alignment_for(image):
    corners = np.float32([[0, 0], [1599, 0], [1599, 1599], [0, 1599]])
    return align_tray(image, corners)


def test_annotation_follows_aperture_without_changing_classifier_crop():
    image = np.full((1600, 1600, 3), 20, np.uint8)
    blank_alignment = alignment_for(image)
    slot = extract_slots(blank_alignment, 'packaged')[3]
    actual = np.round(slot.normalized_center).astype(int) + [6, 4]
    cv2.circle(image, tuple(actual), 8, (160, 160, 160), -1)
    alignment = alignment_for(image)
    slots = extract_slots(alignment, 'packaged')
    slot = slots[3]
    located = AnnotationLayout(alignment, slots, 'packaged').locate(slot, 'TLens')
    assert np.linalg.norm(np.asarray(located.normalized_center) - actual) < 1
    assert located.crop is slot.crop
    assert located.row == slot.row and located.column == slot.column


def test_no_aperture_uses_calibrated_grid_instead_of_spurious_crop_center():
    alignment = alignment_for(np.zeros((1600, 1600, 3), np.uint8))
    slots = extract_slots(alignment, 'bare')
    slot = slots[5]
    shifted = replace(slot, normalized_center=(slot.normalized_center[0] + 10, slot.normalized_center[1] + 8))
    layout = AnnotationLayout(alignment, slots, 'bare')
    located = layout.locate(shifted, 'TLens')
    assert np.allclose(located.normalized_center, slot.normalized_center)
    assert layout.locate(slots[0], 'Excluded') is slots[0]


def test_visible_wells_correct_grid_scale_and_offset():
    image = np.zeros((1600, 1600, 3), np.uint8)
    blank_alignment = alignment_for(image)
    initial = extract_slots(blank_alignment, 'bare')
    for slot in initial:
        x, y = slot.normalized_center
        x, y = round(x + 2), round(y * 1.004 + 1)
        cv2.rectangle(image, (x - 23, y - 23), (x + 23, y + 23), (100, 100, 100), 2)
    alignment = alignment_for(image)
    slots = extract_slots(alignment, 'bare')
    layout = AnnotationLayout(alignment, slots, 'bare')
    assert layout.grid_fitted
    slot = slots[-1]
    located = layout.locate(slot, 'Empty')
    expected = np.array([slot.normalized_center[0] + 2, slot.normalized_center[1] * 1.004 + 1])
    assert np.linalg.norm(np.asarray(located.normalized_center) - expected) < 1.5
