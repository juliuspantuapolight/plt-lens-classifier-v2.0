# Initial MobileNetV3-Small evaluation

Measured locally on 2026-10-18 using CPU FP32, four PyTorch threads, batch size
32, and a confidence threshold of 0.8. Each MobileNet checkpoint was fine-tuned
for one epoch from TorchVision ImageNet weights. The supplied configurations
allow longer runs. The two-class model has 1,519,906 parameters and a 6.2 MB
checkpoint; the existing DINOv2 checkpoints are about 88.3 MB.

## Bare trays

Test set: `data/slots-bare-centered/test`, containing 2,000 crops from four capture
sessions. Training, validation, and test session names do not overlap.

| Model | Correct / total | Missed lenses | False detections | Uncertain | Mean of tray median times |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 bare | 2,000 / 2,000 | 0 | 0 | 0 | 27.07 s |
| MobileNet 224 | 2,000 / 2,000 | 0 | 0 | 0 | 1.62 s |
| MobileNet 128 | 1,989 / 2,000 | 0 | 0 | 11 | 0.72 s |

MobileNet 224 was about **16.7× faster** for complete inspection on the two timed
images. Timings include image loading, alignment, aperture recentering,
classification, quality checks, and rendering; they exclude model loading and
writing output files. Each image was warmed up and measured three times. The
images were `O911A-2638008-001.jpg` (18 occupied slots) and
`A91XB-2626009-002.jpg` (240 occupied slots), both from test sessions.

The 128 variant uses the 224-trained weights with a smaller inference input,
without additional training. Validation crops were used to assess input sizes:
224 had 2 confident errors and 4 uncertain predictions; 128 had no confident
errors and 7 uncertain predictions. At 96, there were 7 confident errors and 118
uncertain predictions, so that size was excluded from test evaluation. Validation
sweep timing ran alongside packaged training and is not used for speed claims.

Use **MobileNet 224** as the first bare-tray deployment candidate. The 128 variant
increases review requirements. The current CLI defaults still select DINOv2;
select a MobileNet folder explicitly with `--model-dir`.

Local artifacts:

- `models/tlens-mobilenet-bare`: recommended 224 candidate.
- `models/tlens-mobilenet-bare-128`: experimental inference resize variant.
- `outputs/model-comparison/mobilenet-bare.json`: complete test results and timings.
- `outputs/model-comparison/mobilenet-input-sizes.json`: validation input-size sweep.
- `outputs/mobilenet-inspection/`: sample annotated image and JSON report.

These results measure the existing annotation-derived labels. They do not establish
accuracy under new lighting, cameras, tray positions, or capture sessions beyond
the available test set. Full-tray counts in the benchmark are model predictions;
they are not independent manual ground truth.

## Packaged trays

Test set: `data/slots-v2/test`, containing 1,275 crops from two capture sessions.
Training, validation, and test session names do not overlap. The MobileNet
checkpoint reached 100% validation top-1 accuracy after its first epoch.

| Model | Correct / total | Missed lenses | False detections | Uncertain | Median tray time |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 v2 | 1,269 / 1,275 | 6 | 0 | 0 | 16.55 s |
| MobileNet 224 | 1,269 / 1,275 | 5 | 0 | 1 | 0.78 s |

Both models scored 99.53% with uncertain predictions counted as incorrect.
MobileNet was about **21.1× faster** on the timed packaged test image,
`A501B-2633011-001.jpg`. Both returned 181 occupied and 74 empty positions on
that full-tray image. Full-frame inspection was used without aperture
recentering, with identical CPU settings and three timed runs after warmup.

The packaged checkpoint is `models/tlens-mobilenet`; raw results are saved in
`outputs/model-comparison/mobilenet-packaged.json`. MobileNet 224 is a reasonable
replacement candidate based on this comparison, but neither model eliminates
missed lenses on the existing packaged test set. Review those failures before
setting production acceptance criteria.

## Reproduce

Train the initial checkpoints:

```bash
python -m src.mobile_train --config configs/mobilenet-bare.yaml --epochs 1
python -m src.mobile_train --config configs/mobilenet.yaml --epochs 1
```

The bare training run was stopped after the first saved epoch; its completed
checkpoint, history, and effective one-epoch training configuration are stored
locally. The 128 folder copies its weights and changes the inference image size
in `slot_model.json`; it was not separately trained.

Use the comparison command in the README for bare trays. For packaged trays:

```bash
python scripts/compare_slot_models.py \
  --model-dirs models/tlens-dinov2-v2 models/tlens-mobilenet \
  --dataset-dir data/slots-v2/test \
  --images assets/2026-10-07-02-00-33-2633011/A501B-2633011-001.jpg \
  --tray-type packaged --full-frame --cpu-threads 4 --repeats 3 \
  --output outputs/model-comparison/mobilenet-packaged.json
```

Run timing comparisons without concurrent training or other benchmarks. These
measurements use the same four PyTorch threads for every model; they are not a
search for each model's optimal thread count. Model weights, datasets, and raw
outputs stay local because those folders are ignored by Git.

Verification: 11 tests passed, including checkpoint loading, RGB normalization,
partial-batch prediction, DINOv2 compatibility, geometry, and output naming. A real
bare-tray MobileNet inspection also produced matching JPG and JSON artifacts.
