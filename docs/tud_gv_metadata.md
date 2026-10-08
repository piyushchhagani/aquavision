# TUD-GV AquaVision Dataset Metadata

## Dataset

TUD-GV Floating Litter Dataset

## Source

Zenodo:
https://zenodo.org/records/13730228

## Dataset Statistics

- Total images: 1,501
- Total bounding boxes: 8,181
- Classes: 1
- Class 0: litter

## Split

- Training: 1,050 images
- Validation: 225 images
- Test: 226 images
- Random seed: 42

## Annotation

Format: YOLO bounding boxes

Each annotation:

class_id x_center y_center width height

Coordinates are normalized to [0, 1].

## Data Integrity

- Image/label matching verified
- 1,501 images
- 1,501 label files
- 8,181 bounding boxes
- 0 invalid annotations
- 0 missing labels
- 0 unmatched labels
- Train/validation/test overlap: 0

## Initial Taxonomy

| ID | Class |
|---:|---|
| 0 | litter |

## Directory

data/
├── raw/tud_gv/
├── interim/tud_gv/
└── processed/tud_gv/

## Important

The original raw dataset is not committed to GitHub.

Model weights and generated outputs are also excluded from Git.
