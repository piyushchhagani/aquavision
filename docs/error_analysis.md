# AquaVision — Baseline Error Analysis

## Evaluation setup

- Model: YOLO11n pretrained baseline
- Dataset: TUD-GV Floating Litter
- Detection class: litter
- Matching method: confidence-ordered greedy one-to-one matching
- IoU matching threshold: 0.50
- Confidence candidates: 0.25, 0.40, 0.50

## Validation threshold comparison

| Confidence | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1228 | 122 | 35 | 0.9096 | 0.9723 | 0.9399 |
| 0.40 | 1213 | 78 | 50 | 0.9396 | 0.9604 | 0.9499 |
| 0.50 | 1195 | 45 | 68 | 0.9637 | 0.9462 | 0.9549 |

## Operating threshold

A confidence threshold of 0.50 is provisionally selected because it achieved the highest
validation F1 among the three tested candidates. This choice favors a stronger
precision-recall balance and fewer false positives than the 0.25 setting.

For applications where missing litter is more costly than reviewing false alarms,
a lower threshold may be more appropriate.

## Test-set threshold comparison

The following results were observed during exploratory test-set analysis:

| Confidence | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1152 | 90 | 30 | 0.9275 | 0.9746 | 0.9505 |
| 0.40 | 1142 | 57 | 40 | 0.9525 | 0.9662 | 0.9593 |
| 0.50 | 1135 | 40 | 47 | 0.9660 | 0.9602 | 0.9631 |

Because multiple thresholds were compared on the test set, these test results should be
treated as exploratory rather than as an unbiased final estimate for the selected
operating threshold. Future threshold selection should use validation data.

## Error investigation

- A per-image evaluator was created to record TP, FP and FN.
- Images with frequent errors were ranked for visual inspection.
- Overlapping prediction pairs with IoU >= 0.50 were collected for review.
- Overlapping predictions are not automatically duplicates: separate litter objects
  can overlap in an image.
- One inspected image, exp55_196.jpg, contained 8 TP, 2 FP and 2 FN at confidence
  0.25 and IoU threshold 0.50.

## Limitations

- This is a single-class detector trained and evaluated on TUD-GV.
- Confidence threshold comparisons depend on the matching method and IoU threshold.
- The custom evaluator uses greedy matching; its metrics are not identical to
  Ultralytics mAP metrics.
- Environmental generalization has not yet been established using an independent
  external dataset.
- Severity estimation and pollution coverage require separate validated methods.
