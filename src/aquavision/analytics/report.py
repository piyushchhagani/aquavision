
"""Aggregate AquaVision litter-detection results and export reports."""

import csv
import json
from pathlib import Path
from statistics import mean


def analyze_results(results):
    """Aggregate per-image inference dictionaries.

    Bounding-box area ratios are detection-box statistics, not true
    pollution coverage. Severity is a configurable preliminary heuristic.
    """
    results = list(results)
    image_count = len(results)
    detection_counts = [int(item.get("detection_count", 0)) for item in results]
    detections = [
        detection
        for item in results
        for detection in item.get("detections", [])
    ]
    confidences = [
        float(d["confidence"])
        for d in detections
        if "confidence" in d
    ]

    total_detections = len(detections)
    average_count = mean(detection_counts) if detection_counts else 0.0
    average_confidence = mean(confidences) if confidences else 0.0

    # Preliminary heuristic based on detected objects per image;
    # not an environmental or regulatory classification.
    if average_count < 1:
        severity = "LOW"
    elif average_count < 5:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return {
        "image_count": image_count,
        "total_detections": total_detections,
        "average_detections_per_image": round(average_count, 4),
        "average_confidence": round(average_confidence, 6),
        "minimum_confidence": round(min(confidences), 6) if confidences else None,
        "maximum_confidence": round(max(confidences), 6) if confidences else None,
        "preliminary_severity": severity,
        "severity_basis": (
            "Average detected litter objects per image: "
            "LOW < 1, MEDIUM >= 1 and < 5, HIGH >= 5. "
            "This is an unvalidated heuristic, not an environmental standard."
        ),
        "images": [
            {
                "image": item.get("image"),
                "detection_count": int(item.get("detection_count", 0)),
                "inference_wall_time_ms": item.get("inference_wall_time_ms"),
                "sum_bbox_area_ratio_percent": item.get(
                    "sum_bbox_area_ratio_percent"
                ),
            }
            for item in results
        ],
    }


def export_report(results, output_dir):
    """Write aggregate JSON and per-image CSV reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)
    summary = analyze_results(results)

    json_path = output_dir / "analytics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    csv_path = output_dir / "per_image_analytics.csv"
    columns = [
        "image",
        "detection_count",
        "inference_wall_time_ms",
        "sum_bbox_area_ratio_percent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary["images"])

    return {"summary": summary, "json_path": str(json_path), "csv_path": str(csv_path)}
