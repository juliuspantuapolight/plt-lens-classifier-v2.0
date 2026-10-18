"""Decode the fixed sticker region and extract its lot identifier."""
import re

import cv2
import numpy as np

from src.tray_geometry import TRAY_CONFIGS, grid_centers


def lot_from_value(value: str) -> str:
    value = value.strip()
    parts = value.split("-")
    return parts[1] if len(parts) == 3 and all(part.strip() for part in parts) else value


def lot_folder_name(lot_number: str | None) -> str:
    if not lot_number:
        return "_unreadable"
    # Keep decoded text as metadata, but use one portable directory component.
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', lot_number).strip().rstrip('. ')
    if not name or name in {".", ".."}:
        name = "_"
    if name.split('.')[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        name = "_" + name
    return name[:120]


def read_sticker_qr(image, alignment, tray_type):
    config = TRAY_CONFIGS[tray_type]
    excluded = set(config.get("excluded", ())) | set(config.get("always_empty", ()))
    grid = grid_centers(tray_type, alignment.normalized.shape)
    dx, dy = grid[1][2] - grid[0][2], grid[config["columns"]][3] - grid[0][3]
    centers = np.array([[x, y] for row, column, x, y in grid if (row, column) in excluded])
    left, top = centers.min(axis=0) - [dx / 2, dy / 2]
    right, bottom = centers.max(axis=0) + [dx / 2, dy / 2]
    corners = np.float32([[[left, top], [right, top], [right, bottom], [left, bottom]]])
    source = cv2.perspectiveTransform(corners, alignment.inverse_matrix)[0]
    width = max(1, round(max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3]))))
    height = max(1, round(max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1]))))
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (width, height))
    detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    attempts = [gray, cv2.copyMakeBorder(gray, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)]
    for candidate in attempts:
        value, _, _ = detector.detectAndDecode(candidate)
        if value.strip():
            return {"qr_value": value, "qr_status": "decoded", "lot_number": lot_from_value(value)}
    return {"qr_value": None, "qr_status": "unreadable", "lot_number": None}
