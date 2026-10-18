"""Tray alignment, configurable grid localization, and slot extraction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


TRAY_CONFIGS = {
    "packaged": {"rows": 17, "columns": 15, "left": 0.09, "right": 0.91, "top": 0.075, "bottom": 0.915,
                 "always_empty": ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))},
    "bare": {"rows": 20, "columns": 20, "left": 0.08, "right": 0.92, "top": 0.079, "bottom": 0.918,
             "excluded": tuple((row, column) for row in range(1, 7) for column in range(1, 5))},
}


@dataclass(frozen=True)
class Alignment:
    normalized: np.ndarray
    corners: np.ndarray
    forward_matrix: np.ndarray
    inverse_matrix: np.ndarray


@dataclass(frozen=True)
class Slot:
    row: int
    column: int
    normalized_center: tuple[float, float]
    original_center: tuple[float, float]
    crop: np.ndarray


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return ordered


def _detect_saturated_tray_corners(image: np.ndarray, minimum_area_ratio: float) -> np.ndarray | None:
    """Return a rectangular boundary for a strongly colored tray, if present."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 70, 80], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    candidates = [contour for contour in contours if cv2.contourArea(contour) >= image_area * minimum_area_ratio]
    if not candidates:
        return None
    return _order_corners(cv2.boxPoints(cv2.minAreaRect(max(candidates, key=cv2.contourArea))))


def detect_tray_corners(image: np.ndarray, minimum_area_ratio: float = 0.2) -> np.ndarray:
    """Find a colored-tray or edge-defined quadrilateral boundary or raise ValueError."""
    saturated_corners = _detect_saturated_tray_corners(image, minimum_area_ratio)
    if saturated_corners is not None:
        return saturated_corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    candidates = [contour for contour in contours if cv2.contourArea(contour) >= image_area * minimum_area_ratio]
    if not candidates:
        raise ValueError("Tray boundary was not detected; provide manual corners or review the image.")
    contour = max(candidates, key=cv2.contourArea)
    approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    points = approximation.reshape(-1, 2) if len(approximation) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    return _order_corners(points)


def align_tray(image: np.ndarray, corners: np.ndarray | None = None, output_size: tuple[int, int] = (1600, 1600)) -> Alignment:
    corners = detect_tray_corners(image) if corners is None else _order_corners(corners)
    width, height = output_size
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    forward = cv2.getPerspectiveTransform(corners, destination)
    inverse = cv2.getPerspectiveTransform(destination, corners)
    return Alignment(cv2.warpPerspective(image, forward, output_size), corners, forward, inverse)


def grid_centers(tray_type: str, image_shape: tuple[int, ...], geometry: dict | None = None) -> list[tuple[int, int, float, float]]:
    if tray_type not in TRAY_CONFIGS:
        raise ValueError(f"Unsupported tray type {tray_type!r}; select one of {sorted(TRAY_CONFIGS)}")
    config = {**TRAY_CONFIGS[tray_type], **(geometry or {})}
    height, width = image_shape[:2]; rows, columns = config["rows"], config["columns"]
    xs = np.linspace(config["left"] * width, config["right"] * width, columns)
    ys = np.linspace(config["top"] * height, config["bottom"] * height, rows)
    return [(row + 1, column + 1, float(x), float(y)) for row, y in enumerate(ys) for column, x in enumerate(xs)]


def extract_slots(alignment: Alignment, tray_type: str, crop_scale: float = 0.72, geometry: dict | None = None) -> list[Slot]:
    centers = grid_centers(tray_type, alignment.normalized.shape, geometry)
    if len(centers) != TRAY_CONFIGS[tray_type]["rows"] * TRAY_CONFIGS[tray_type]["columns"]:
        raise ValueError("Grid generation produced an unexpected slot count.")
    width = abs(centers[1][2] - centers[0][2]) if TRAY_CONFIGS[tray_type]["columns"] > 1 else 50
    height = abs(centers[TRAY_CONFIGS[tray_type]["columns"]][3] - centers[0][3]) if TRAY_CONFIGS[tray_type]["rows"] > 1 else 50
    half_width, half_height = max(1, round(width * crop_scale / 2)), max(1, round(height * crop_scale / 2))
    inverse_points = cv2.perspectiveTransform(np.asarray([[[x, y] for _, _, x, y in centers]], dtype=np.float32), alignment.inverse_matrix)[0]
    slots = []
    for (row, column, x, y), original in zip(centers, inverse_points):
        crop = alignment.normalized[max(0, round(y) - half_height):round(y) + half_height, max(0, round(x) - half_width):round(x) + half_width]
        slots.append(Slot(row, column, (x, y), (float(original[0]), float(original[1])), crop))
    return slots


def recenter_slot_apertures(alignment: Alignment, slots: list[Slot], tray_type: str) -> list[Slot]:
    """Recenter slot crops on a nearby circular lens aperture when one is visible."""
    columns = TRAY_CONFIGS[tray_type]["columns"]
    horizontal = abs(slots[1].normalized_center[0] - slots[0].normalized_center[0])
    vertical = abs(slots[columns].normalized_center[1] - slots[0].normalized_center[1])
    maximum_offset = min(horizontal, vertical) * 0.5
    recentered = []
    for slot in slots:
        crop_height, crop_width = slot.crop.shape[:2]
        center_x, center_y = slot.normalized_center
        search_half_width, search_half_height = round(horizontal * 0.6), round(vertical * 0.6)
        left, right = max(0, round(center_x) - search_half_width), min(alignment.normalized.shape[1], round(center_x) + search_half_width)
        top, bottom = max(0, round(center_y) - search_half_height), min(alignment.normalized.shape[0], round(center_y) + search_half_height)
        search = alignment.normalized[top:bottom, left:right]
        grayscale = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        if tray_type == "bare":
            grayscale = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(grayscale)
            filtered = cv2.GaussianBlur(grayscale, (5, 5), 0)
            circle_settings = {"minDist": 8, "param1": 40, "param2": 5, "minRadius": 7, "maxRadius": 22}
        else:
            filtered = cv2.medianBlur(grayscale, 5)
            circle_settings = {"minDist": 15, "param1": 80, "param2": 12, "minRadius": 7, "maxRadius": 28}
        circles = cv2.HoughCircles(filtered, cv2.HOUGH_GRADIENT, 1.0, **circle_settings)
        if circles is None:
            recentered.append(slot); continue
        candidate = min(circles[0], key=lambda circle: (circle[0] + left - center_x) ** 2 + (circle[1] + top - center_y) ** 2)
        detected_center = (float(candidate[0] + left), float(candidate[1] + top))
        if np.linalg.norm(np.asarray(detected_center) - np.asarray(slot.normalized_center)) > maximum_offset:
            recentered.append(slot); continue
        x, y = (round(value) for value in detected_center)
        crop = alignment.normalized[max(0, y - crop_height // 2):y + crop_height // 2, max(0, x - crop_width // 2):x + crop_width // 2]
        original = cv2.perspectiveTransform(np.asarray([[[detected_center[0], detected_center[1]]]], dtype=np.float32), alignment.inverse_matrix)[0][0]
        recentered.append(Slot(slot.row, slot.column, detected_center, (float(original[0]), float(original[1])), crop))
    return recentered


def draw_grid_debug(image: np.ndarray, slots: list[Slot], tray_type: str) -> np.ndarray:
    output = image.copy()
    for slot in slots:
        cv2.circle(output, tuple(round(value) for value in slot.original_center), 4, (255, 0, 255), -1)
    cv2.putText(output, f"{tray_type}: {len(slots)} slots", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2)
    return output
