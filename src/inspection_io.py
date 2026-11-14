"""Output naming helpers for tray inspection artifacts."""
import csv
from datetime import datetime
from pathlib import Path

from src.sticker_qr import lot_folder_name


def inspection_output_paths(output_dir: Path, image_path: Path, lot_number: str | None = None) -> tuple[Path, Path]:
    output_dir = output_dir / lot_folder_name(lot_number)
    stem = f"{image_path.stem}_inspection"
    return output_dir / f"{stem}.jpg", output_dir / f"{stem}.json"


def write_slot_csv(result: dict, csv_path: Path) -> None:
    """One row per classified slot: tray_name,tx(column),ty(row),lens(1=TLens),flipped,timestamp."""
    tray_name = result.get("qr_value") or Path(result["image"]).stem
    timestamp = str(datetime.now())
    slots = sorted((slot for slot in result["slots"] if slot["prediction"] != "Excluded"),
                   key=lambda slot: (slot["row"], slot["column"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tray_name", "tx", "ty", "lens", "flipped", "timestamp"])
        # shortcut: flipped is always 0, the model has no flip class; Uncertain counts as no lens.
        writer.writerows([tray_name, slot["column"], slot["row"], int(slot["prediction"] == "TLens"), 0, timestamp] for slot in slots)


def collect_input_images(input_dir: Path, output_dir: Path) -> list[Path]:
    """Collect clean images recursively, excluding generated annotation files."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    output_root = output_dir.resolve()
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
        and not path.stem.lower().endswith(("_annotated", "-annotated", "_inspection", "-inspection"))
        and not path.resolve().is_relative_to(output_root)
    )
