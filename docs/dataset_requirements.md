# AquaVision Dataset Requirements

## Primary Objective

Build a computer vision dataset suitable for detecting visible pollution on water surfaces.

## Required Characteristics

The primary dataset should preferably contain:

- Water-surface imagery
- Floating or visible surface pollution
- Object-level annotations
- Bounding boxes suitable for object detection
- Multiple pollution categories where available
- Sufficient image diversity
- Different viewpoints and environments
- Different lighting conditions
- Real-world images
- Clear dataset license
- Research and academic usage compatibility

## Preferred Annotation Formats

Priority:

1. YOLO
2. COCO
3. Pascal VOC
4. Other formats that can be reliably converted

## Segmentation

Segmentation annotations are highly valuable but are not mandatory for the initial detection MVP.

If available, preserve them for future segmentation experiments.

## Dataset Diversity

We should evaluate:

- Lakes
- Rivers
- Ponds
- Canals
- Reservoirs
- Urban water bodies
- Rural water bodies
- Different weather conditions
- Different lighting conditions
- Different camera distances
- Different pollution densities

## Dataset Quality Checks

Before training, AquaVision must check:

- Corrupted images
- Missing annotations
- Invalid annotations
- Duplicate images
- Empty annotations
- Incorrect bounding boxes
- Unsupported image formats
- Extreme image dimensions
- Class imbalance

## Dataset Split

Initial target:

- Training: 70%
- Validation: 15%
- Testing: 15%

The final test set must remain isolated from model training and hyperparameter tuning.

## Local Dataset

A separate local dataset may later be collected for real-world generalization testing.

Local test images must not be mixed into the training set without explicit dataset versioning.

## Dataset Selection Principle

Do not select a dataset only because it has a large number of images.

Relevance to water-surface pollution, annotation quality, diversity, licensing, and research value are more important than raw image count.
