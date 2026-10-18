"""Create a debug image of aligned tray slot centers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tray_geometry import TRAY_CONFIGS, align_tray, draw_grid_debug, extract_slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tray-type", choices=tuple(TRAY_CONFIGS), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/grid_debug.jpg"))
    parser.add_argument("--full-frame", action="store_true", help="Use image borders as manual tray corners for calibration/debugging.")
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None: raise SystemExit(f"Unreadable image: {args.image}")
    try:
        height, width = image.shape[:2]
        corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32) if args.full_frame else None
        alignment = align_tray(image, corners=corners); slots = extract_slots(alignment, args.tray_type)
    except ValueError as error:
        raise SystemExit(f"Alignment review required: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), draw_grid_debug(image, slots, args.tray_type))
    print(f"Wrote {args.output} with {len(slots)} slot centers.")


if __name__ == "__main__": main()
