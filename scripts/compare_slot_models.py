"""Compare classifiers on held-out crops and optional full-tray inputs."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.slot_inference import SlotInspector


def evaluate(inspector, dataset):
    records = [(path, label) for label in ("Empty", "TLens") for path in sorted((dataset / label).glob("*.jpg"))]
    if not records or len({label for _, label in records}) != 2:
        raise ValueError("Held-out dataset must contain both classes")
    counts = {"correct": 0, "missed_lenses": 0, "false_detections": 0, "uncertain": 0}
    confusion = {label: {prediction: 0 for prediction in ("Empty", "TLens", "Uncertain")} for label in ("Empty", "TLens")}
    inspector.classify([cv2.imread(str(records[0][0]))])  # untimed warmup
    started = time.perf_counter()
    for start in range(0, len(records), inspector.batch_size):
        batch = records[start:start + inspector.batch_size]
        crops = [cv2.imread(str(path)) for path, _ in batch]
        if any(crop is None for crop in crops):
            raise ValueError("Unreadable held-out crop")
        for (_, truth), (prediction, confidence) in zip(batch, inspector.classify(crops)):
            if confidence < inspector.confidence_threshold:
                prediction = "Uncertain"
            confusion[truth][prediction] += 1
            counts["correct"] += prediction == truth
            counts["missed_lenses"] += truth == "TLens" and prediction == "Empty"
            counts["false_detections"] += truth == "Empty" and prediction == "TLens"
            counts["uncertain"] += prediction == "Uncertain"
    return {**counts, "total": len(records), "accuracy": counts["correct"] / len(records),
            "confusion_matrix": confusion, "seconds_including_crop_reads": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Held-out split, e.g. data/slots-bare-centered/test")
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    parser.add_argument("--tray-type", choices=("bare", "packaged"), required=True)
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument("--recenter-apertures", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence", type=float, default=.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/model-comparison/mobilenet.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    results = []
    for directory in args.model_dirs:
        print(f"Evaluating {directory} on {args.dataset_dir}", file=sys.stderr, flush=True)
        inspector = SlotInspector(directory, args.confidence, args.batch_size, cpu_threads=args.cpu_threads)
        result = {"model_dir": str(directory), "architecture": inspector.architecture,
                  "dataset_dir": str(args.dataset_dir), "tray_type": args.tray_type,
                  "full_frame": args.full_frame, "recenter_apertures": args.recenter_apertures,
                  "repeats": args.repeats, "device": str(inspector.device), "cpu_threads": torch.get_num_threads(),
                  "batch_size": args.batch_size, "confidence_threshold": args.confidence,
                  "held_out": evaluate(inspector, args.dataset_dir), "trays": []}
        for image in args.images:
            print(f"Timing {directory}: {image.name}", file=sys.stderr, flush=True)
            inspector.inspect(image, args.tray_type, args.full_frame, 100, args.recenter_apertures)
            seconds = []; classification = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                report, _ = inspector.inspect(image, args.tray_type, args.full_frame, 100, args.recenter_apertures)
                seconds.append(time.perf_counter() - started)
                classification.append(report["timings_seconds"]["classification"])
            result["trays"].append({"image": str(image), "median_seconds": statistics.median(seconds),
                "median_classification_seconds": statistics.median(classification),
                "occupied": report["occupied_count"], "empty": report["empty_count"], "uncertain": report["uncertain_count"]})
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        del inspector


if __name__ == "__main__":
    main()
