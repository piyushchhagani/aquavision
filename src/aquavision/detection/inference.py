
"""Reusable image inference for AquaVision."""

from pathlib import Path
import json
import time

import cv2
from ultralytics import YOLO


class AquaVisionDetector:
    """Detect litter and return machine-readable image-level results."""

    def __init__(self, weights_path, confidence=0.5, image_size=640, device=None):
        self.weights_path = str(weights_path)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device
        self.model = YOLO(self.weights_path)

    def predict(self, image_path, output_dir, confidence=None):
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]
        effective_confidence = (
            self.confidence if confidence is None else float(confidence)
        )
        if not 0.01 <= effective_confidence <= 0.99:
            raise ValueError('confidence must be between 0.01 and 0.99')

        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        result = self.model.predict(
            source=str(image_path),
            conf=effective_confidence,
            iou=0.5,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        wall_time_ms = (time.perf_counter() - start) * 1000

        detections = []
        total_bbox_area = 0.0

        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().tolist())
            confidence = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())

            x1, x2 = sorted((
                max(0.0, min(x1, width)),
                max(0.0, min(x2, width)),
            ))
            y1, y2 = sorted((
                max(0.0, min(y1, height)),
                max(0.0, min(y2, height)),
            ))

            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            total_bbox_area += bbox_area

            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(confidence, 6),
                "bbox_xyxy": [round(v, 2) for v in (x1, y1, x2, y2)],
                "bbox_area_pixels": round(bbox_area, 2),
                "bbox_image_area_ratio_percent": round(
                    100.0 * bbox_area / (width * height), 6
                ),
            })

        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        result.save(filename=str(annotated_path))

        data = {
            "image": image_path.name,
            "image_width": width,
            "image_height": height,
            "model": Path(self.weights_path).name,
            "confidence_threshold": effective_confidence,
            "iou_threshold": 0.5,
            "detection_count": len(detections),
            "detections": detections,
            "sum_bbox_area_ratio_percent": round(
                100.0 * total_bbox_area / (width * height), 6
            ),
            "inference_wall_time_ms": round(wall_time_ms, 3),
            "annotated_image": str(annotated_path),
            "note": (
                "Summed bounding-box area is not water-pollution coverage "
                "and may double-count overlapping boxes."
            ),
        }

        json_path = output_dir / f"{image_path.stem}_results.json"
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        data["results_json"] = str(json_path)

        return data
