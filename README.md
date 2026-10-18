# TLens Slot Inspector

This project inspects fixed-position TLens trays using OpenCV tray alignment and DINOv2 image classifiers. Each slot is classified as `TLens`, `Empty`, or `Uncertain`.

Supported tray types:

- `packaged`: 17 rows x 15 columns (255 positions)
- `bare`: 20 rows x 20 columns (400 positions)

## Project layout

- `src/tray_geometry.py` aligns trays and extracts fixed slot crops.
- `src/slot_train.py` fine-tunes the DINOv2 slot classifier.
- `src/slot_inference.py` produces per-slot classifications, counts, annotated images, and JSON results.
- `data/slots/` contains the train, validation, and test slot crops.
- `models/tlens-dinov2/` contains the trained classifier.

## Inspect one tray

Activate the virtual environment, then run:

```powershell
python -m src.slot_inference `
  --model-dir models\tlens-dinov2 `
  --image "assets\<session>\<tray-image>.jpg" `
  --tray-type packaged `
  --full-frame `
  --output-dir outputs\dinov2-inspection
```

The command writes matching `<image>_inspection.jpg` and `<image>_inspection.json` files to the output directory. Use `--full-frame` for the current camera framing; automatic tray-boundary alignment can be calibrated later.

## Train or continue training

```powershell
python -m src.slot_train --config configs\dinov2.yaml
```

For GPU training, run the same command in a Kaggle notebook after placing the slot dataset at `data/slots` and saving the resulting model files in `models/tlens-dinov2`.

## Prepare and train the bare tray model

```powershell
python scripts\prepare_slot_dataset.py `
  --assets-dir assets\bare `
  --output-dir data\slots-bare-centered `
  --tray-type bare `
  --full-frame `
  --center-tlens-on-markers `
  --min-marker-count 0 `
  --balanced-session-splits `
  --recenter-apertures

python -m src.slot_train --config configs\dinov2-bare.yaml
```

Inspect a bare tray with `--tray-type bare`, `--model-dir models\tlens-dinov2-bare`, and `--recenter-apertures`.

## Faster CPU inspection on Linux

Activate the existing environment with `source .venv/bin/activate`. Use optional
dynamic INT8 inference to reduce CPU work without retraining or modifying model
files:

```bash
python -m src.slot_inference \
  --model-dir models/tlens-dinov2-bare \
  --image "assets/bare/2026-10-01-02-07-20-2443004/M921B-2443004-001.jpg" \
  --tray-type bare \
  --full-frame \
  --recenter-apertures \
  --cpu-int8 \
  --summary \
  --output-dir outputs/bare-inspection-int8
```

INT8 is CPU-only and can change confidence scores or predictions. Compare it with
the default FP32 mode on representative images before production use. Omit
`--cpu-int8` to retain FP32 inference and automatic CUDA selection.

For multiple trays, replace `--image ...` with `--input-dir "assets/bare/<session>"`
to load the model once. This processes every JPG/PNG in that directory, so use a
directory containing only clean input images, without annotation overlays.

`--summary` prints one line per image; saved JSON reports still contain all slots.
Each report includes `timings_seconds` for image reading, alignment/cropping,
recentering, classification (including preprocessing), and quality/rendering,
plus the device, precision, batch size, and PyTorch CPU thread count in `runtime`.
Model loading time is printed separately to stderr. `inference_seconds` excludes
model loading and output-file writes.

Use `--cpu-threads N` and `--batch-size N` to benchmark your hardware. Fewer threads
or larger batches are not necessarily faster. Defaults remain unchanged.
