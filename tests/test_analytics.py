import json
from pathlib import Path

from aquavision.analytics.report import analyze_results, export_report


def test_analytics_summary_and_exports(tmp_path):
    results = [
        {
            "image": "image1.jpg",
            "detection_count": 2,
            "detections": [
                {"confidence": 0.9},
                {"confidence": 0.8},
            ],
            "inference_wall_time_ms": 12.5,
            "sum_bbox_area_ratio_percent": 1.2,
        },
        {
            "image": "image2.jpg",
            "detection_count": 0,
            "detections": [],
            "inference_wall_time_ms": 10.0,
            "sum_bbox_area_ratio_percent": 0.0,
        },
    ]

    summary = analyze_results(results)
    assert summary["image_count"] == 2
    assert summary["total_detections"] == 2
    assert summary["average_detections_per_image"] == 1.0
    assert summary["average_confidence"] == 0.85
    assert summary["preliminary_severity"] == "MEDIUM"

    exported = export_report(results, tmp_path)
    assert Path(exported["json_path"]).exists()
    assert Path(exported["csv_path"]).exists()

    saved = json.loads(Path(exported["json_path"]).read_text())
    assert saved["total_detections"] == 2


def test_empty_results():
    summary = analyze_results([])
    assert summary["image_count"] == 0
    assert summary["total_detections"] == 0
    assert summary["average_confidence"] == 0.0
    assert summary["preliminary_severity"] == "LOW"
