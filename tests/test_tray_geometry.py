import numpy as np

import cv2

from src.tray_geometry import TRAY_CONFIGS, detect_tray_corners, grid_centers


def test_packaged_grid_has_expected_positions():
    assert len(grid_centers("packaged", (1600, 1600, 3))) == 255


def test_bare_grid_has_expected_positions():
    assert len(grid_centers("bare", (1600, 1600, 3))) == 400


def test_grid_positions_stay_within_image():
    for _, _, x, y in grid_centers("bare", (100, 200, 3)):
        assert 0 <= x < 200 and 0 <= y < 100


def test_detect_tray_corners_prefers_large_saturated_tray():
    image = np.zeros((500, 600, 3), dtype=np.uint8)
    corners = np.array([[80, 90], [520, 70], [550, 430], [60, 450]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (180, 80, 255))

    detected = detect_tray_corners(image)

    assert np.allclose(detected, corners.astype(np.float32), atol=45)
