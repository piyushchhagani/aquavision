"""Experimental association of litter detections with a water-candidate mask."""

from typing import Any

import numpy as np

from aquavision.water.mask import validate_water_mask


def associate_detections_with_water(
    mask: Any,
    detections: list[dict],
) -> dict:
    """Associate boxes with water using their center pixel.

    This is a spatial heuristic, not pollution segmentation.
    """
    water = validate_water_mask(mask)
    height, width = water.shape
    associated = []

    for detection in detections:
        box = (
            detection.get("bbox_xyxy")
            or detection.get("bbox")
            or detection.get("box")
        )
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = map(float, box)
        cx = max(0, min(width - 1, int((x1 + x2) / 2)))
        cy = max(0, min(height - 1, int((y1 + y2) / 2)))

        item = dict(detection)
        item["center_inside_water_candidate"] = bool(water[cy, cx])
        associated.append(item)

    inside = sum(
        item["center_inside_water_candidate"] for item in associated
    )

    return {
        "water_candidate_pixel_percent": round(float(water.mean() * 100), 2),
        "detection_count": len(detections),
        "detections_with_valid_boxes": len(associated),
        "detection_centers_inside_water_candidate": int(inside),
        "association_method": "bounding-box center point",
        "warning": (
            "HSV water mask and center-point association are experimental. "
            "These values do not measure pollution area or validated water coverage."
        ),
        "detections": associated,
    }
