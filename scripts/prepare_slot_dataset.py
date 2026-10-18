"""Build clean TLens/Empty slot crops from paired reference overlays."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tray_geometry import Alignment, TRAY_CONFIGS, align_tray, extract_slots, recenter_slot_apertures


def pairs(assets: Path) -> list[tuple[Path, Path, str]]:
    result = []
    for overlay in assets.rglob("*"):
        if not overlay.is_file() or overlay.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if overlay.stem.endswith("_annotated"):
            original_stem = overlay.stem.removesuffix("_annotated")
        elif overlay.stem.endswith("-annotated"):
            original_stem = overlay.stem.removesuffix("-annotated")
        else:
            continue
        original = next((overlay.with_name(f"{original_stem}{suffix}")
                         for suffix in (".jpg", ".jpeg", ".png")
                         if overlay.with_name(f"{original_stem}{suffix}").exists()), None)
        if original is not None:
            result.append((original, overlay, overlay.parent.name))
    return sorted(result)


def split_sessions(sessions: list[str], seed: int) -> dict[str, set[str]]:
    values = sorted(set(sessions)); random.Random(seed).shuffle(values)
    train, val = max(1, round(len(values) * .7)), max(1, round(len(values) * .15))
    return {"train": set(values[:train]), "val": set(values[train:train + val]), "test": set(values[train + val:])}


def balanced_session_splits(session_counts: dict[str, tuple[int, int]]) -> dict[str, set[str]]:
    """Keep sessions isolated while balancing size and TLens rate across splits."""
    sessions = sorted(session_counts)
    total_slots = sum(total for total, _ in session_counts.values())
    total_tlens = sum(tlens for _, tlens in session_counts.values())
    target_slots = total_slots * 0.15
    target_rate = total_tlens / total_slots
    candidates = []
    for size in range(1, min(4, len(sessions) - 2) + 1):
        for subset in itertools.combinations(sessions, size):
            slots = sum(session_counts[name][0] for name in subset)
            if not target_slots * 0.55 <= slots <= target_slots * 1.5:
                continue
            tlens = sum(session_counts[name][1] for name in subset)
            score = abs(slots - target_slots) / target_slots + 3 * abs(tlens / slots - target_rate)
            candidates.append((score, set(subset)))
    valid_pairs = [
        (validation[0] + test[0], validation[1], test[1])
        for validation in candidates
        for test in candidates
        if validation[1].isdisjoint(test[1])
    ]
    if not valid_pairs:
        return split_sessions(sessions, 42)
    _, validation, test = min(valid_pairs, key=lambda item: item[0])
    return {"train": set(sessions) - validation - test, "val": validation, "test": test}


def marker_session_counts(records: list[tuple[Path, Path, str]], positions: int) -> dict[str, tuple[int, int]]:
    counts = defaultdict(lambda: [0, 0])
    for _, overlay_path, session in records:
        overlay = cv2.imread(str(overlay_path))
        if overlay is None:
            continue
        counts[session][0] += positions
        counts[session][1] += len(green_marker_centers(overlay))
    return {session: (values[0], values[1]) for session, values in counts.items()}


def has_green_marker(image) -> bool:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return int(cv2.countNonZero(cv2.inRange(hsv, (40, 80, 80), (90, 255, 255)))) >= 30


def green_marker_centers(image: np.ndarray) -> list[tuple[float, float]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            centers.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    return centers


def marker_slot_centers(slots, markers: list[tuple[float, float]], columns: int) -> dict[int, tuple[float, float]]:
    if not markers:
        return {}
    grid = np.asarray([slot.normalized_center for slot in slots], dtype=np.float32)
    horizontal = abs(grid[1, 0] - grid[0, 0])
    vertical = abs(grid[columns, 1] - grid[0, 1])
    maximum_distance = min(horizontal, vertical) * 0.75
    matches: dict[int, tuple[float, tuple[float, float]]] = {}
    for marker in markers:
        distances = np.linalg.norm(grid - marker, axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance <= maximum_distance and (index not in matches or distance < matches[index][0]):
            matches[index] = (distance, marker)
    return {index: marker for index, (_, marker) in matches.items()}


def crop_at(image: np.ndarray, center: tuple[float, float], shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    x, y = (round(value) for value in center)
    return image[max(0, y - height // 2):y + height // 2, max(0, x - width // 2):x + width // 2]


def has_centered_aperture(crop: np.ndarray) -> bool:
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cv2.medianBlur(grayscale, 5), cv2.HOUGH_GRADIENT, 1.0, 15,
                               param1=80, param2=12, minRadius=7, maxRadius=28)
    if circles is None:
        return False
    center = np.array([(crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2])
    maximum_offset = min(crop.shape[:2]) * 0.1
    return any(np.linalg.norm(np.asarray(circle[:2]) - center) <= maximum_offset for circle in circles[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/slots"))
    parser.add_argument("--tray-type", choices=tuple(TRAY_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument("--min-tlens-dark-ratio", type=float, default=0.0)
    parser.add_argument("--max-empty-dark-ratio", type=float, default=1.0)
    parser.add_argument("--center-tlens-on-markers", action="store_true")
    parser.add_argument("--min-marker-count", type=int, default=10)
    parser.add_argument("--require-centered-aperture", action="store_true")
    parser.add_argument("--balanced-session-splits", action="store_true")
    parser.add_argument("--recenter-apertures", action="store_true")
    args = parser.parse_args(); records = pairs(args.assets_dir)
    if not records: raise SystemExit("No clean/reference image pairs found.")
    positions = TRAY_CONFIGS[args.tray_type]["rows"] * TRAY_CONFIGS[args.tray_type]["columns"]
    split_map = balanced_session_splits(marker_session_counts(records, positions)) if args.balanced_session_splits else split_sessions([session for _, _, session in records], args.seed)
    always_empty = set(TRAY_CONFIGS[args.tray_type].get("always_empty", ()))
    manifest, summary = [], defaultdict(lambda: defaultdict(int))
    for original_path, overlay_path, session in records:
        clean, overlay = cv2.imread(str(original_path)), cv2.imread(str(overlay_path))
        if clean is None or overlay is None: continue
        height, width = clean.shape[:2]
        overlay_height, overlay_width = overlay.shape[:2]
        if (overlay_height, overlay_width) != (height, width):
            clean_ratio, overlay_ratio = width / height, overlay_width / overlay_height
            if not np.isclose(clean_ratio, overlay_ratio, rtol=0.01):
                summary["skipped"]["annotation_aspect_ratio_mismatch"] += 1; continue
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
        corners = None
        if args.full_frame:
            corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        try:
            alignment = align_tray(clean, corners=corners)
        except ValueError:
            summary["skipped"]["alignment_failed"] += 1; continue
        overlay_normalized = cv2.warpPerspective(overlay, alignment.forward_matrix, (alignment.normalized.shape[1], alignment.normalized.shape[0]))
        overlay_alignment = Alignment(overlay_normalized, alignment.corners, alignment.forward_matrix, alignment.inverse_matrix)
        split = next(name for name, values in split_map.items() if session in values)
        base_slots = extract_slots(alignment, args.tray_type)
        clean_slots = recenter_slot_apertures(alignment, base_slots, args.tray_type) if args.recenter_apertures else base_slots
        overlay_slots = extract_slots(overlay_alignment, args.tray_type)
        marker_centers = marker_slot_centers(
            base_slots,
            green_marker_centers(overlay_normalized),
            TRAY_CONFIGS[args.tray_type]["columns"],
        ) if args.center_tlens_on_markers else {}
        if args.center_tlens_on_markers and len(marker_centers) < args.min_marker_count:
            summary["skipped"]["insufficient_green_markers"] += 1; continue
        for index, (clean_slot, overlay_slot) in enumerate(zip(clean_slots, overlay_slots)):
            label = "Empty" if (clean_slot.row, clean_slot.column) in always_empty else ("TLens" if index in marker_centers else "Empty")
            use_marker_center = label == "TLens" and index in marker_centers and not args.recenter_apertures
            crop = crop_at(alignment.normalized, marker_centers[index], clean_slot.crop.shape) if use_marker_center else clean_slot.crop
            grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_ratio = float((grayscale < 70).mean())
            if label == "TLens" and dark_ratio < args.min_tlens_dark_ratio:
                summary["skipped"]["tlens_crop_without_dark_housing"] += 1; continue
            if label == "Empty" and dark_ratio > args.max_empty_dark_ratio:
                summary["skipped"]["empty_crop_with_dark_housing"] += 1; continue
            if label == "TLens" and args.require_centered_aperture and not has_centered_aperture(crop):
                summary["skipped"]["tlens_crop_without_centered_aperture"] += 1; continue
            destination = args.output_dir / split / label / f"{session}_{original_path.stem}_r{clean_slot.row:02d}_c{clean_slot.column:02d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(destination), crop)
            manifest.append({"file": str(destination.relative_to(args.output_dir)), "source_image": str(original_path), "session": session, "tray_type": args.tray_type, "row": clean_slot.row, "column": clean_slot.column, "label": label})
            summary[split][label] += 1
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
