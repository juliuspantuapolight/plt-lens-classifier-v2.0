"""Apply manual slot corrections to an inspection JSON and redraw its annotated image.

    python -m src.corrections outputs/.../tray_inspection.json --set 3,4=Empty --set 7,1=TLens

Without --set, the result is only marked as reviewed. The classifier's original answer is kept in
`model_prediction` on every slot whose label differs from it.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.inspection_io import write_slot_csv
from src.inspection_rendering import AnnotationLayout
from src.tray_geometry import align_tray, extract_slots, full_frame_corners, recenter_slot_apertures

EDITABLE_LABELS = ("TLens", "Empty")


def parse_correction(text: str) -> tuple[tuple[int, int], str]:
    try:
        position, label = text.split("=")
        row, column = (int(part) for part in position.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected ROW,COLUMN=LABEL, got {text!r}") from None
    if label not in EDITABLE_LABELS:
        raise argparse.ArgumentTypeError(f"Label must be one of {EDITABLE_LABELS}, got {label!r}")
    return (row, column), label


def _write_atomically(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def apply_corrections(json_path: Path, corrections: dict[tuple[int, int], str]) -> dict:
    result = json.loads(json_path.read_text(encoding="utf-8"))
    records = {(slot["row"], slot["column"]): slot for slot in result["slots"]}
    for position, label in corrections.items():
        record = records.get(position)
        if record is None or record["prediction"] == "Excluded":
            raise ValueError(f"Slot {position} is not a classified position on this {result['tray_type']} tray.")
        if label not in EDITABLE_LABELS:
            raise ValueError(f"Label must be one of {EDITABLE_LABELS}, got {label!r}")
        original = record.get("model_prediction", record["prediction"])
        record["prediction"] = label
        if label == original:
            record.pop("model_prediction", None)
        else:
            record["model_prediction"] = original

    image = cv2.imread(result["image"])
    if image is None:
        raise ValueError(f"Source image is needed to redraw annotations but cannot be read: {result['image']}")
    tray_type = result["tray_type"]
    # shortcut: results saved before these keys existed assume run.py defaults; rerun them if they used other flags.
    full_frame = result.get("full_frame", True)
    recenter = result.get("recenter_apertures", tray_type == "bare")
    alignment = align_tray(image, corners=full_frame_corners(image) if full_frame else None)
    slots = extract_slots(alignment, tray_type)
    if recenter:
        slots = recenter_slot_apertures(alignment, slots, tray_type)
    layout = AnnotationLayout(alignment, slots, tray_type)
    rendered = image.copy()
    for slot in slots:
        record = records[(slot.row, slot.column)]
        label = record["prediction"]
        located = layout.locate(slot, label)
        layout.draw(rendered, located, label)
        record["center_x"], record["center_y"] = (round(value, 2) for value in located.original_center)
        result["occupancy_matrix"][slot.row - 1][slot.column - 1] = label

    labels = [record["prediction"] for record in result["slots"]]
    result["occupied_count"], result["empty_count"], result["uncertain_count"] = (
        labels.count("TLens"), labels.count("Empty"), labels.count("Uncertain"))
    result["corrected_count"] = sum("model_prediction" in record for record in result["slots"])
    result["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # A person checked it; only open questions (uncertain slots, unknown lot) keep it in review.
    result["inspection_status"] = "reviewed" if not result["uncertain_count"] and result["qr_status"] == "decoded" else "review_required"

    ok, encoded = cv2.imencode(Path(result["annotated_image"]).suffix or ".jpg", rendered)
    if not ok:
        raise RuntimeError("Could not encode the annotated image.")
    _write_atomically(Path(result["annotated_image"]), encoded.tobytes())
    _write_atomically(json_path, json.dumps(result, indent=2).encode("utf-8"))
    write_slot_csv(result, json_path.with_suffix(".csv"))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--set", dest="corrections", type=parse_correction, action="append", default=[],
                        metavar="ROW,COLUMN=LABEL", help="TLens or Empty; repeat for more slots")
    args = parser.parse_args(argv)
    if not args.json_path.is_file():
        parser.error(f"Result file not found: {args.json_path}")
    result = apply_corrections(args.json_path, dict(args.corrections))
    print(f"Saved: TLens={result['occupied_count']} Empty={result['empty_count']} Uncertain={result['uncertain_count']} "
          f"corrected={result['corrected_count']} status={result['inspection_status']}")


if __name__ == "__main__":
    main()
