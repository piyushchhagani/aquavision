import importlib.util
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "aquavision_dashboard", Path("apps/dashboard/app.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_detection_rows():
    results = [{
        "image": "sample.jpg",
        "detections": [
            {"confidence": 0.91, "class_name": "litter", "bbox": [1, 2, 3, 4]}
        ],
    }]
    rows = MODULE.detection_rows(results)
    assert len(rows) == 1
    assert rows[0]["Image"] == "sample.jpg"
    assert rows[0]["Confidence"] == 0.91
    assert rows[0]["Class"] == "litter"


def test_image_rows():
    rows = MODULE.image_rows([{
        "image": "sample.jpg",
        "detection_count": 2,
        "detections": [{"confidence": 0.8}, {"confidence": 0.6}],
        "inference_wall_time_ms": 12.5,
    }])
    assert rows[0]["Detections"] == 2
    assert rows[0]["Average confidence (%)"] == 70.0


def test_exports():
    rows = [{"Image": "sample.jpg", "Detections": 2}]
    assert b"sample.jpg" in MODULE.csv_bytes(rows)
    assert b'"Detections": 2' in MODULE.json_bytes({"Detections": 2})


def test_empty_results():
    assert MODULE.detection_rows([]) == []
    assert MODULE.image_rows([]) == []
