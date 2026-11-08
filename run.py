"""Inspect a tray with the recommended MobileNet defaults."""
import argparse
from pathlib import Path

from src.slot_inference import main as inspect_main

MODEL_DIRS = {"bare": Path("models/tlens-mobilenet-bare"), "packaged": Path("models/tlens-mobilenet")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tray", nargs="?", choices=("auto", "bare", "packaged"), default="auto",
                        help="Tray type; omit (or 'auto') to detect it per image")
    parser.add_argument("input", type=Path, help="Image or folder (includes subfolders; skips annotations and inspection outputs)")
    parser.add_argument("--model-dir", type=Path, help="Override the MobileNet model folder (requires an explicit tray type)")
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
    if args.model_dir and args.tray == "auto":
        parser.error("--model-dir needs an explicit tray type (bare or packaged).")
    model_dirs = MODEL_DIRS if args.tray == "auto" else {args.tray: args.model_dir or MODEL_DIRS[args.tray]}
    for model_dir in model_dirs.values():
        if not model_dir.is_dir():
            parser.error(f"Model folder not found: {model_dir} "
                         "(models/ is not in git; copy the trained model folder there or pass --model-dir)")
    command = ["--tray-type", args.tray,
               "--input-dir" if args.input.is_dir() else "--image", str(args.input),
               "--output-dir", str(args.output_dir), "--cpu-threads", str(args.cpu_threads),
               "--batch-size", str(args.batch_size), "--confidence", str(args.confidence),
               "--blur-threshold", str(args.blur_threshold), "--summary"]
    if args.tray == "auto":
        for tray, model_dir in model_dirs.items():
            command += [f"--{tray}-model-dir", str(model_dir)]
    else:
        command += ["--model-dir", str(model_dirs[args.tray])]
    if args.full_frame:
        command.append("--full-frame")
    if args.recenter_apertures:
        command.append("--recenter-apertures")
    elif args.recenter_apertures is None and args.tray != "packaged":
        command += ["--recenter-apertures", "bare"]  # bare trays only, also under auto
    inspect_main(command)


if __name__ == "__main__":
    main()
