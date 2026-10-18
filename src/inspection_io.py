"""Output naming helpers for tray inspection artifacts."""
from pathlib import Path


def inspection_output_paths(output_dir: Path, image_path: Path) -> tuple[Path, Path]:
    stem = f"{image_path.stem}_inspection"
    return output_dir / f"{stem}.jpg", output_dir / f"{stem}.json"
