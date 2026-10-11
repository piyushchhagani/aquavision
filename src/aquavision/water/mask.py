"""Validation and coverage utilities for binary water masks."""

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
