
"""FastAPI endpoints for AquaVision litter detection."""

import json
import os
import time
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from pydantic import BaseModel

from aquavision.analytics.report import analyze_results
from aquavision.detection.inference import AquaVisionDetector
from aquavision.water.mask import segment_water_hsv, calculate_water_coverage
from aquavision.water.analysis import associate_detections_with_water


PROJECT_ROOT = Path(__file__).resolve().parents[3]

app = FastAPI(
    title="AquaVision API",
    description="Detect visible floating litter in water-surface images.",
    version="0.1.0",
)

_detector = None
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def get_output_dir() -> Path:
    path = Path(
        os.getenv(
            "AQUAVISION_OUTPUT_DIR",
            str(PROJECT_ROOT / "outputs" / "api_results"),
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_detector():
    global _detector

    if _detector is None:
        weights_env = os.getenv("AQUAVISION_WEIGHTS")
        if weights_env:
            weights_path = Path(weights_env)
        else:
            default_weights = PROJECT_ROOT / "models" / "tud_gv_best.pt"
            fallback_weights = PROJECT_ROOT / "models" / "yolo11n.pt"
            weights_path = default_weights if default_weights.is_file() else fallback_weights

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. Place trained weights at models/tud_gv_best.pt or specify AQUAVISION_WEIGHTS."
            )

        _detector = AquaVisionDetector(
            weights_path=weights_path,
            confidence=0.35,
            image_size=640,
        )

    return _detector


@app.get("/health")
def health():
    return {"status": "ok", "service": "AquaVision API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = Form(0.35, ge=0.01, le=0.99)):
    request_started = time.perf_counter()
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 15 MB.",
        )

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        Path(file.filename or "upload").stem,
    ).strip("._") or "upload"

    request_id = uuid.uuid4().hex[:12]
    output_root = get_output_dir()
    request_dir = output_root / f"{safe_name}_{request_id}"
    request_dir.mkdir(parents=True, exist_ok=False)

    input_path = request_dir / f"input{ALLOWED_TYPES[file.content_type]}"
    input_path.write_bytes(content)

    try:
        detector = get_detector()
        result = detector.predict(
            image_path=input_path,
            output_dir=request_dir,
            confidence=confidence,
        )

        # Experimental HSV mask for visualization only; not validated segmentation.
        import cv2
        import numpy as np

        source_image = cv2.imread(str(input_path))
        water_mask = segment_water_hsv(source_image)
        water_coverage = calculate_water_coverage(water_mask)
        water_analysis = associate_detections_with_water(
            water_mask, result.get('detections', [])
        )
        mask_path = request_dir / "water_mask.png"
        overlay_path = request_dir / "water_mask_overlay.jpg"
        cv2.imwrite(str(mask_path), water_mask.astype(np.uint8) * 255)
        overlay = source_image.copy()
        tint = np.zeros_like(source_image)
        tint[:, :, 1] = 220
        blended = cv2.addWeighted(source_image, 0.60, tint, 0.40, 0)
        overlay[water_mask] = blended[water_mask]
        cv2.imwrite(str(overlay_path), overlay)

        # Return stable artifact paths after prediction completes.
        result["image"] = file.filename or input_path.name
        result["annotated_image"] = str(
            Path(result["annotated_image"]).resolve()
        )
        result["results_json"] = str(
            Path(result["results_json"]).resolve()
        )
        result["input_image"] = str(input_path.resolve())
        result["water_mask"] = str(mask_path.resolve())
        result["water_mask_overlay"] = str(overlay_path.resolve())
        result["water_mask_coverage_percent"] = round(water_coverage, 2)
        result["water_mask_method"] = "experimental HSV thresholding"
        result["water_mask_warning"] = "Experimental color threshold only; not validated water segmentation or pollution coverage."
        result["water_analysis"] = water_analysis

        result["api_processing_time_ms"] = round(
            (time.perf_counter() - request_started) * 1000, 3
        )
        result["confidence_threshold"] = confidence

        (request_dir / "api_response.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc
    finally:
        await file.close()


class AnalyzeRequest(BaseModel):
    results: list[dict]


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Aggregate existing prediction results into summary analytics."""
    try:
        return analyze_results(request.results)
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prediction results: {exc}",
        ) from exc
