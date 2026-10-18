"""Inspect a tray with the recommended MobileNet defaults."""
import argparse
from pathlib import Path

from src.slot_inference import main as inspect_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tray", choices=("bare", "packaged"))
    parser.add_argument("input", type=Path, help="Image or folder (includes subfolders; skips annotations and inspection outputs)")
    parser.add_argument("--model-dir", type=Path, help="Override the MobileNet model folder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mobilenet-inspection"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence", type=float, default=.8)
    parser.add_argument("--blur-threshold", type=float, default=100.)
    parser.add_argument("--full-frame", action=argparse.BooleanOptionalAction, default=True,
                        help="Use current camera framing; --no-full-frame detects tray boundaries")
    parser.add_argument("--recenter-apertures", action=argparse.BooleanOptionalAction, default=None,
                        help="Defaults to enabled for bare trays")
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    model_dir = args.model_dir or Path("models") / (
        "tlens-mobilenet-bare" if args.tray == "bare" else "tlens-mobilenet")
    command = ["--tray-type", args.tray, "--model-dir", str(model_dir),
               "--input-dir" if args.input.is_dir() else "--image", str(args.input),
               "--output-dir", str(args.output_dir), "--cpu-threads", str(args.cpu_threads),
               "--batch-size", str(args.batch_size), "--confidence", str(args.confidence),
               "--blur-threshold", str(args.blur_threshold), "--summary"]
    if args.full_frame:
        command.append("--full-frame")
    recenter = args.recenter_apertures if args.recenter_apertures is not None else args.tray == "bare"
    if recenter:
        command.append("--recenter-apertures")
    inspect_main(command)


if __name__ == "__main__":
    main()
