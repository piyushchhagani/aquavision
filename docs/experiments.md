# AquaVision Experiments

## Experiment 001 — YOLO11n TUD-GV Baseline

### Dataset

- Dataset: TUD-GV Floating Litter
- Images: 1,501
- Classes: 1 (`litter`)
- Train: 1,050
- Validation: 225
- Test: 226
- Test ground-truth boxes: 1,182
- Split seed: 42

### Model

- Architecture: YOLO11n
- Pretrained: Yes
- Image size: 640
- Epochs: 30
- Batch size: 16
- Device: NVIDIA Tesla T4
- Seed: 42
- Deterministic: Yes

### Purpose

Establish the first reproducible object-detection baseline for AquaVision.

### Test Evaluation

| Metric | Result |
|---|---:|
| Precision | 0.9735 |
| Recall | 0.9611 |
| mAP@50 | 0.9848 |
| mAP@50-95 | 0.7757 |

### Test Inference Speed

- Preprocess: 4.0 ms/image
- Inference: 5.5 ms/image
- Postprocess: 3.9 ms/image

### Error Analysis

To be completed using IoU-based prediction-to-ground-truth matching.

- False positives:
- False negatives:
- Small-object failures:
- Dense-object failures:
- Background/reflection confusion:
- Low-confidence detections:

### Conclusion

The YOLO11n baseline provides a strong reference point for
single-class floating-litter detection on the TUD-GV test set.

Future experiments must be compared against this result.
