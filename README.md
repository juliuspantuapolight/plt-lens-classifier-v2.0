# TLens Slot Inspector

This project inspects fixed-position TLens trays using OpenCV tray alignment and MobileNet or DINOv2 image classifiers. Each usable slot is classified as `TLens`, `Empty`, or `Uncertain`; sticker positions are `Excluded`.

Supported tray types:

- `packaged`: 17 rows x 15 columns (255 positions)
- `bare`: 20 rows x 20 columns (400 positions)

## Project layout

- `src/tray_geometry.py` aligns trays and extracts fixed slot crops.
- `src/slot_train.py` fine-tunes the DINOv2 slot classifier.
- `src/slot_inference.py` produces per-slot classifications, counts, annotated images, and JSON results.
- `data/slots/` contains the train, validation, and test slot crops.
- `models/tlens-dinov2/` contains the trained classifier.

## Quick inspection

From the project root, activate the environment once:

```bash
source .venv/bin/activate
```

Then provide the tray type and image:

```bash
python run.py bare "assets/bare/2026-10-01-02-05-13-2638008/O911A-2638008-001.jpg"
python run.py packaged "assets/2026-10-07-02-00-33-2633011/A501B-2633011-001.jpg"
```

You can also pass a directory to process multiple images with one model load:

```bash
python run.py bare "assets/bare"
python run.py packaged "assets/2026-10-07-02-00-12-2633006"
```

Folder mode scans all subfolders and loads the model once. It skips files ending
in `_annotated`, `-annotated`, `_inspection`, or `-inspection`, and skips the
output directory. Supported extensions: JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP
(case insensitive). Choose a folder containing one tray type.

Each image is saved under its decoded lot folder. Repeated input basenames get
a stable path suffix to prevent overwriting each other's results. An empty
folder reports an error before loading the model.

The runner selects the corresponding MobileNet model, uses full-frame alignment,
enables aperture recentering for bare trays, uses four CPU threads, and prints a
short summary. Matching `<image>_inspection.jpg` and `<image>_inspection.json`
files are saved in `outputs/mobilenet-inspection/<lot-number>/`.

Annotations follow the reference samples: green rings mark lenses, red boxes
mark empty positions, and yellow boxes mark uncertain positions. Sticker areas
are left unmarked and skipped by the classifier: rows 1–3 / columns 1–2 for
packaged trays, and rows 1–6 / columns 1–4 for bare trays, using the current
fixed camera calibration.

JSON retains the complete rectangular grid. Sticker records have
`prediction: "Excluded"` and `confidence: null`; `excluded_count` reports them
separately. `classified_positions` is 249 for packaged trays and 376 for bare
trays. Occupied, empty, and uncertain counts sum to `classified_positions`,
while `total_positions` remains 255 or 400 respectively. Counts are shown in the
CLI summary and JSON, leaving the annotated image free of header text.

The sticker QR code determines the lot folder. A three-part value such as
`A501B-2633020-003` uses its middle part, `2633020`. Other values use the whole
decoded name. JSON includes `qr_value`, `qr_status`, and `lot_number`.
For example, the pair is saved as:

```text
outputs/mobilenet-inspection/2633020/A501B-2633020-003_inspection.jpg
outputs/mobilenet-inspection/2633020/A501B-2633020-003_inspection.json
```

Unreadable QR codes go to `_unreadable/`, with `lot_number: null` and
`inspection_status: "review_required"`; the filename is not used to guess the
lot. Characters unsuitable for directory names are replaced with underscores
in the folder name while preserving the decoded lot in JSON. Previous outputs
saved directly in the output directory are retained; new runs use lot folders.

Annotation centers are refined separately from classification crops. Visible
round apertures guide lens rings, and a robust fit to visible rectangular wells
corrects small grid offsets, scale changes, and skew on bare trays. Detection is
bounded to the assigned slot; faint or ambiguous features fall back to the
calibrated grid. This adjustment does not change classifier inputs or counts.
JSON `center_x` / `center_y` describe the drawn annotation center;
`crop_center_x` / `crop_center_y` retain the center used for classification.

Optional settings remain available:

```bash
python run.py bare "path/to/image.jpg" --output-dir outputs/my-inspection
python run.py bare "path/to/image.jpg" --confidence 0.9 --cpu-threads 8
python run.py bare "path/to/image.jpg" --no-full-frame --no-recenter-apertures
python run.py --help
```

The full `python -m src.slot_inference` command is also available for advanced
options, including DINOv2 INT8 inference.

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
to load the model once. This scans subfolders too and skips annotation files and inspection outputs.

`--summary` prints one line per image; saved JSON reports still contain all slots.
Each report includes `timings_seconds` for image reading, alignment/cropping,
recentering, classification (including preprocessing), and quality/rendering,
plus the device, precision, batch size, and PyTorch CPU thread count in `runtime`.
Model loading time is printed separately to stderr. `inference_seconds` excludes
model loading and output-file writes.

Use `--cpu-threads N` and `--batch-size N` to benchmark your hardware. Fewer threads
or larger batches are not necessarily faster. Defaults remain unchanged.

## Lightweight MobileNetV3-Small classifier

MobileNet uses the existing slot datasets and tray inspection pipeline. DINOv2
models remain supported; the inspector automatically recognizes MobileNet model
folders containing `slot_model.json` and `model.pt`.

Train the bare model (ImageNet weights download on the first run):

```bash
python -m src.mobile_train --config configs/mobilenet-bare.yaml
```

Use `configs/mobilenet.yaml` for packaged trays and `data/slots-v2`. Both configs
start at 224×224, fine-tune all layers, and save the checkpoint with the lowest
validation loss. Use `--epochs 1` for an initial training run. Set `model.image_size` to 128 or 96 and choose a distinct
`model.output_dir` to train smaller-input variants. Select variants using the
validation split before evaluating the final choice on the test split.

```bash
python -m src.slot_inference \
  --model-dir models/tlens-mobilenet-bare \
  --image assets/bare/2026-10-01-02-05-13-2638008/O911A-2638008-001.jpg \
  --tray-type bare --full-frame --recenter-apertures \
  --cpu-threads 4 --summary --output-dir outputs/mobilenet-inspection
```

MobileNet currently runs FP32. `--cpu-int8` applies only to DINOv2; dynamically
quantizing linear layers would leave MobileNet's convolutional backbone FP32.

Compare held-out crops and repeated full-tray timings with identical settings:

```bash
python scripts/compare_slot_models.py \
  --model-dirs models/tlens-dinov2-bare models/tlens-mobilenet-bare \
  --dataset-dir data/slots-bare-centered/test \
  --images assets/bare/2026-10-01-02-05-13-2638008/O911A-2638008-001.jpg \
  --tray-type bare --full-frame --recenter-apertures \
  --cpu-threads 4 --repeats 3
```

The comparison saves `outputs/model-comparison/mobilenet.json`, including a
confusion matrix, missed lenses, false detections, uncertain predictions, and
median full-tray and classification timings after warmup. Accuracy counts
uncertain predictions as incorrect; missed lenses and false detections count
confident errors separately. Full-tray timing excludes model loading and output
writes. Tray counts are predictions, not ground-truth accuracy measurements.

Initial local measurements and model recommendations are recorded in
[docs/mobilenet-results.md](docs/mobilenet-results.md). On the existing bare-tray
test set, MobileNet at 224 matched DINOv2 on all 2,000 crops and reduced complete
inspection from approximately 27.1 to 1.6 seconds per tray on the two timed
images. Local trained model folders and raw benchmark outputs are ignored by Git.
The packaged MobileNet candidate matched DINOv2 v2 at 99.53% on 1,275 test crops,
with 5 missed lenses and 1 uncertain result versus DINOv2's 6 misses. Its timed
packaged inspection took about 0.78 seconds versus 16.55 seconds for DINOv2.
