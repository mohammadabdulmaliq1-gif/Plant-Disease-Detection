from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from PIL import UnidentifiedImageError
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..models import PredictionResponse, ErrorResponse
from ..model_service import model_service
from ..config import settings


# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file content type."""
    # Check content type - be lenient to handle browser variations
    allowed_types = [
        "image/jpeg", "image/jpg", "image/png", "image/webp",
        "image/bmp", "application/octet-stream",
    ]
    # Some browsers send content_type as None or ''
    if file.content_type and file.content_type not in allowed_types:
        # Allow if content_type starts with 'image/'
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP, BMP"
            )


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(..., description="Plant image file (JPEG, PNG, WEBP)")
):
    """Predict plant disease from an uploaded image.

    - Upload a photo of a plant leaf
    - Returns the predicted disease, confidence score, and treatment recommendations
    """
    try:
        print(f"[DEBUG] Received file: name={image.filename!r}, "
              f"content_type={image.content_type!r}, "
              f"size={getattr(image, 'size', 'unknown')}")

        # Validate model is loaded
        if not model_service.is_healthy():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model service is not ready. "
                    "The model may still be loading — please wait a moment and try again."
                )
            )

        # Validate image file content type
        validate_image_file(image)

        # Read image bytes
        image_bytes = await image.read()

        # Validate size after reading
        if len(image_bytes) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_image_size_bytes / (1024*1024):.0f} MB"
            )

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        print(f"[DEBUG] Image bytes received: {len(image_bytes):,} bytes")

        # Make prediction
        result = model_service.predict(image_bytes)

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except (UnidentifiedImageError, ValueError) as e:
        print(f"[WARNING] Invalid image content: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unreadable image file: {str(e)}"
        )
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        try:
            await image.close()
        except Exception:
            pass