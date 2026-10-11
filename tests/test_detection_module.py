from pathlib import Path
import numpy as np
import torch

from aquavision.detection.inference import AquaVisionDetector


class FakeBox:
    xyxy = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
    conf = torch.tensor([0.9])
    cls = torch.tensor([0])


class FakeResult:
    names = {0: "litter"}
    boxes = [FakeBox()]

    def save(self, filename):
        import cv2
        assert cv2.imwrite(filename, np.zeros((100, 100, 3), dtype=np.uint8))


class FakeModel:
    def predict(self, **kwargs):
        return [FakeResult()]


def test_predict_returns_detection_and_writes_outputs(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"mock image")

    monkeypatch.setattr(
        "aquavision.detection.inference.cv2.imread",
        lambda path: np.zeros((100, 100, 3), dtype=np.uint8),
    )

    detector = object.__new__(AquaVisionDetector)
    detector.weights_path = "fake-best.pt"
    detector.confidence = 0.5
    detector.image_size = 640
    detector.device = "cpu"
    detector.model = FakeModel()

    result = detector.predict(image_path, tmp_path / "out")

    assert result["detection_count"] == 1
    assert result["detections"][0]["class_name"] == "litter"
    assert result["detections"][0]["confidence"] == 0.9
    assert result["detections"][0]["bbox_xyxy"] == [10.0, 20.0, 40.0, 60.0]
    assert result["detections"][0]["bbox_area_pixels"] == 1200.0
    assert result["annotated_image"]
    assert Path(result["annotated_image"]).exists()
    assert Path(result["results_json"]).exists()
