
"""Water-mask validation and coverage utilities.

HSV segmentation is experimental and must not be treated as
reliable water or pollution segmentation.
"""

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_water_mask(mask: ArrayLike) -> NDArray[np.bool_]:
    """Validate a 2D binary mask encoded as 0/1 or 0/255."""
    array = np.asarray(mask)

    if array.size == 0:
        raise ValueError("Water mask must not be empty.")
    if array.ndim != 2:
        raise ValueError("Water mask must be a 2D array.")
    if not (
        np.issubdtype(array.dtype, np.number)
        or array.dtype == np.bool_
    ):
        raise ValueError("Water mask must contain numeric or boolean values.")
    if not np.isfinite(array).all():
        raise ValueError("Water mask must not contain NaN or infinite values.")
    if not np.isin(np.unique(array), [0, 1, 255]).all():
        raise ValueError("Water mask must be binary, encoded as 0/1 or 0/255.")

    return array != 0


def calculate_water_coverage(mask: ArrayLike) -> float:
    """Return the percentage of pixels marked as water."""
    return float(validate_water_mask(mask).mean() * 100.0)


def segment_water_hsv(image: ArrayLike) -> NDArray[np.bool_]:
    """Create an experimental HSV water-candidate mask."""
    array = np.asarray(image)

    if array.size == 0:
        raise ValueError("Image must not be empty.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image must be a 3-channel BGR image.")
    if array.dtype != np.uint8:
        raise ValueError("Image must use uint8 pixel values.")

    hsv = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)

    lower = np.array([65, 30, 25], dtype=np.uint8)
    upper = np.array([115, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask > 0
