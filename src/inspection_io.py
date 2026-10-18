"""Output naming helpers for tray inspection artifacts."""
from pathlib import Path

from src.sticker_qr import lot_folder_name


def inspection_output_paths(output_dir: Path, image_path: Path, lot_number: str | None = None) -> tuple[Path, Path]:
    output_dir = output_dir / lot_folder_name(lot_number)
    stem = f"{image_path.stem}_inspection"
    return output_dir / f"{stem}.jpg", output_dir / f"{stem}.json"


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
