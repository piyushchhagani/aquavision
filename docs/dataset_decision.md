# AquaVision Primary Dataset Decision

## Primary Dataset

TUD-GV Dataset for Floating Litter Detection

Source:
https://zenodo.org/records/13730228

Reason:

- Specifically designed for floating litter detection
- Freshwater environment
- Object detection annotations
- 1,501 annotated images
- 8,181 litter annotations
- Direct relevance to AquaVision's MVP
- Suitable for YOLO-based baseline detection

## Supplementary Datasets

### FloW
Floating waste detection in inland waters.

### IWHR_AI_Lable_Floater_V1
3,000 images of floating debris in real inland-water scenarios.

### TACO
General litter dataset for future robustness experiments.

### WSODD
Water-surface object detection dataset with a rubbish category and additional contextual water-surface objects.

## Initial Detection Taxonomy

The first model will use the original annotation taxonomy.

For the TUD-GV floating-litter detection subset:

- class 0: litter

No additional classes will be invented.

## Future Taxonomy

Fine-grained categories such as plastic, paper, metal, glass, textile, organic debris, etc. will only be introduced when supported by actual annotations or a separately annotated AquaVision dataset.

## Independent Generalization Test

A future AquaVision Local Dataset will be collected separately and will not be mixed into training data without explicit dataset versioning.
