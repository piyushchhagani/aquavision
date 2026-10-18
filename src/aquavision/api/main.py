
"""FastAPI endpoints for AquaVision litter detection."""

import json
import os
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from aquavision.analytics.report import analyze_results
from aquavision.detection.inference import AquaVisionDetector


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
        weights_path = Path(
            os.getenv(
                "AQUAVISION_WEIGHTS",
                str(PROJECT_ROOT / "models" / "tud_gv_best.pt"),
            )
        )

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}"
            )

        _detector = AquaVisionDetector(
            weights_path=weights_path,
            confidence=0.5,
            image_size=640,
        )

    return _detector


@app.get("/health")
def health():
    return {"status": "ok", "service": "AquaVision API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        )

        # Return stable artifact paths after prediction completes.
        result["image"] = file.filename or input_path.name
        result["annotated_image"] = str(
            Path(result["annotated_image"]).resolve()
        )
        result["results_json"] = str(
            Path(result["results_json"]).resolve()
        )
        result["input_image"] = str(input_path.resolve())

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
