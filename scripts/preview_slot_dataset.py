"""Create a contact sheet for manual slot-label review."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/slots/train"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/slot_label_preview.jpg"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); randomizer = random.Random(args.seed)
    tiles = []
    for label, color in (("TLens", (0, 255, 0)), ("Empty", (0, 0, 255))):
        paths = sorted((args.dataset_dir / label).glob("*.jpg")); selected = randomizer.sample(paths, min(args.per_class, len(paths)))
        for path in selected:
            image = cv2.imread(str(path)); image = cv2.resize(image, (120, 120))
            cv2.putText(image, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA); tiles.append(image)
    columns = 10; rows = (len(tiles) + columns - 1) // columns
    canvas = np.full((rows * 120, columns * 120, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns); canvas[y * 120:(y + 1) * 120, x * 120:(x + 1) * 120] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(args.output), canvas)
    print(f"Wrote {args.output} with {len(tiles)} labeled crops.")


if __name__ == "__main__": main()
