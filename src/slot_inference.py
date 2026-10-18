"""Full-tray inspection using calibrated geometry and a trained slot classifier."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.image_quality import assess_blur
from src.inspection_io import collect_input_images, inspection_output_paths
from src.inspection_rendering import AnnotationLayout
from src.sticker_qr import read_sticker_qr
from src.tray_geometry import TRAY_CONFIGS, align_tray, extract_slots, recenter_slot_apertures


class SlotInspector:
    def __init__(self, model_dir: str | Path, confidence_threshold: float = .8, batch_size: int = 32,
                 cpu_int8: bool = False, cpu_threads: int | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if cpu_threads is not None:
            if cpu_threads < 1:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(cpu_threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cpu_int8 and self.device.type != "cpu":
            raise ValueError("CPU INT8 requires CPU inference; omit --cpu-int8 when using CUDA.")
        if (Path(model_dir) / "slot_model.json").exists():
            if cpu_int8:
                raise ValueError("--cpu-int8 is currently supported for DINOv2 only.")
            from src.mobile_classifier import load_mobile
            self.processor, self.model = load_mobile(Path(model_dir))
            self.architecture = "mobilenet_v3_small"
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
            self.model = AutoModelForImageClassification.from_pretrained(model_dir)
            self.architecture = "dinov2"
        self.model = self.model.to(self.device).eval()
        if cpu_int8:
            # Convert in memory: trained weights on disk remain unchanged.
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
        self.cpu_int8 = cpu_int8
        self.confidence_threshold, self.batch_size = confidence_threshold, batch_size

    @torch.inference_mode()
    def classify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        results = []
        for start in range(0, len(crops), self.batch_size):
            images = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops[start:start + self.batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().numpy()
            for row in probabilities:
                index = int(row.argmax()); results.append((self.model.config.id2label[index], float(row[index])))
        return results

    def inspect(self, image_path: str | Path, tray_type: str, full_frame: bool, blur_threshold: float,
                recenter_apertures: bool = False, dark_housing_threshold: float | None = None) -> tuple[dict, np.ndarray]:
        timings = {}; started = time.perf_counter()
        path = Path(image_path); image = cv2.imread(str(path))
        if image is None: raise ValueError(f"Unreadable image: {path}")
        timings["image_read"] = time.perf_counter() - started
        started = time.perf_counter()
        height, width = image.shape[:2]; corners = None
        if full_frame:
            corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        alignment = align_tray(image, corners=corners); slots = extract_slots(alignment, tray_type)
        timings["alignment_and_crops"] = time.perf_counter() - started
        started = time.perf_counter()
        sticker = read_sticker_qr(image, alignment, tray_type)
        timings["qr_reading"] = time.perf_counter() - started
        started = time.perf_counter()
        if recenter_apertures:
            slots = recenter_slot_apertures(alignment, slots, tray_type)
        timings["recenter_apertures"] = time.perf_counter() - started
        excluded = set(TRAY_CONFIGS[tray_type].get("excluded", ())) | set(TRAY_CONFIGS[tray_type].get("always_empty", ()))
        started = time.perf_counter()
        predicted = self.classify([slot.crop for slot in slots if (slot.row, slot.column) not in excluded]) if dark_housing_threshold is None else []
        timings["classification"] = time.perf_counter() - started
        started = time.perf_counter()
        iterator = iter(predicted); records = []
        matrix = [["Uncertain" for _ in range(TRAY_CONFIGS[tray_type]["columns"])] for _ in range(TRAY_CONFIGS[tray_type]["rows"])]
        rendered = image.copy()
        annotation_layout = AnnotationLayout(alignment, slots, tray_type)
        for slot in slots:
            if (slot.row, slot.column) in excluded: label, confidence = "Excluded", None
            elif dark_housing_threshold is not None:
                grayscale = cv2.cvtColor(slot.crop, cv2.COLOR_BGR2GRAY)
                dark_ratio = float((grayscale < 70).mean())
                label, confidence = ("TLens", dark_ratio) if dark_ratio >= dark_housing_threshold else ("Empty", 1.0 - dark_ratio)
            else:
                label, confidence = next(iterator)
                if confidence < self.confidence_threshold: label = "Uncertain"
            matrix[slot.row - 1][slot.column - 1] = label
            crop_center = slot.original_center
            slot = annotation_layout.locate(slot, label)
            annotation_layout.draw(rendered, slot, label)
            records.append({"row": slot.row, "column": slot.column, "prediction": label, "confidence": round(confidence, 6) if confidence is not None else None, "center_x": round(slot.original_center[0], 2), "center_y": round(slot.original_center[1], 2), "crop_center_x": round(crop_center[0], 2), "crop_center_y": round(crop_center[1], 2)})
        counts = {label: sum(item["prediction"] == label for item in records) for label in ("TLens", "Empty", "Uncertain", "Excluded")}
        quality = assess_blur(image, blur_threshold)
        result = {"image": str(path), "tray_type": tray_type, "rows": TRAY_CONFIGS[tray_type]["rows"], "columns": TRAY_CONFIGS[tray_type]["columns"], "total_positions": len(records), "classified_positions": len(records) - counts["Excluded"], "excluded_count": counts["Excluded"], "occupied_count": counts["TLens"], "empty_count": counts["Empty"], "uncertain_count": counts["Uncertain"], "inspection_status": "review_required" if counts["Uncertain"] or quality["is_severely_blurred"] else "complete", "image_quality": quality, "dark_housing_threshold": dark_housing_threshold, "slots": records, "occupancy_matrix": matrix}
        result.update(sticker)
        if sticker["qr_status"] != "decoded":
            result["inspection_status"] = "review_required"
        if sum(counts.values()) != len(records): raise RuntimeError("Slot count invariant failed.")
        timings["quality_and_rendering"] = time.perf_counter() - started
        result["timings_seconds"] = {name: round(seconds, 4) for name, seconds in timings.items()}
        result["runtime"] = {"architecture": self.architecture, "device": str(self.device), "precision": "dynamic_int8" if self.cpu_int8 else "float32",
                             "batch_size": self.batch_size, "cpu_threads": torch.get_num_threads()}
        return result, rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model-dir", type=Path, default=Path("models/tlens-dinov2")); parser.add_argument("--image", type=Path); parser.add_argument("--input-dir", type=Path); parser.add_argument("--tray-type", choices=tuple(TRAY_CONFIGS), required=True); parser.add_argument("--output-dir", type=Path, default=Path("outputs/inspection")); parser.add_argument("--confidence", type=float, default=.8); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--blur-threshold", type=float, default=100.); parser.add_argument("--full-frame", action="store_true"); parser.add_argument("--recenter-apertures", action="store_true"); parser.add_argument("--dark-housing-threshold", type=float)
    parser.add_argument("--cpu-int8", action="store_true", help="Use dynamic INT8 linear layers on CPU; validate accuracy before production use.")
    parser.add_argument("--cpu-threads", type=int, help="Override PyTorch CPU threads (positive integer).")
    parser.add_argument("--summary", action="store_true", help="Print counts and timing instead of the full JSON; saved reports remain complete.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("--batch-size and --cpu-threads must be positive.")
    if bool(args.image) == bool(args.input_dir): parser.error("Specify exactly one of --image or --input-dir.")
    try:
        paths = [args.image] if args.image else collect_input_images(args.input_dir, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    if not paths:
        parser.error("No clean input images found in the folder.")
    stem_counts = Counter(path.stem.casefold() for path in paths)
    if args.input_dir:
        print(f"Found {len(paths)} input images in {args.input_dir} (including subfolders).", file=sys.stderr)
    started = time.perf_counter()
    inspector = SlotInspector(args.model_dir, args.confidence, args.batch_size, args.cpu_int8, args.cpu_threads)
    print(f"Model ready in {time.perf_counter() - started:.2f}s on {inspector.device} "
          f"({'INT8' if args.cpu_int8 else 'FP32'}).", file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True); results = []
    for path in paths:
        started = time.perf_counter(); result, rendered = inspector.inspect(path, args.tray_type, args.full_frame, args.blur_threshold, args.recenter_apertures, args.dark_housing_threshold); result["inference_seconds"] = round(time.perf_counter() - started, 4)
        output_name = path
        if stem_counts[path.stem.casefold()] > 1:
            # Distinguish repeated basenames without changing the reported source path.
            relative = path.relative_to(args.input_dir) if args.input_dir else path
            suffix = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12]
            output_name = path.with_stem(f"{path.stem}_{suffix}")
        image_output, json_output = inspection_output_paths(args.output_dir, output_name, result["lot_number"])
        image_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_output), rendered)
        result["annotated_image"] = str(image_output)
        result["result_json"] = str(json_output)
        json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if args.summary:
            print(f"{path.name}: Lot={result['lot_number'] or 'unreadable'} TLens={result['occupied_count']} Empty={result['empty_count']} "
                  f"Uncertain={result['uncertain_count']} Excluded={result['excluded_count']} time={result['inference_seconds']:.2f}s")
        else:
            results.append(result)
    if not args.summary:
        print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
