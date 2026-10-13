
import numpy as np
import pytest

from aquavision.water.mask import (
    validate_water_mask,
    calculate_water_coverage,
    segment_water_hsv,
)


def test_water_coverage_known_mask():
    mask = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    assert calculate_water_coverage(mask) == 37.5


def test_water_coverage_full_mask():
    assert calculate_water_coverage(np.ones((3, 3), dtype=np.uint8)) == 100.0


def test_water_coverage_empty_foreground():
    assert calculate_water_coverage(np.zeros((3, 3), dtype=np.uint8)) == 0.0


def test_255_mask_is_supported():
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    assert calculate_water_coverage(mask) == 50.0


def test_mask_must_be_two_dimensional():
    with pytest.raises(ValueError, match="2D"):
        validate_water_mask(np.zeros((2, 2, 3)))


def test_mask_must_not_be_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_water_mask(np.array([]))


def test_mask_must_be_binary():
    with pytest.raises(ValueError, match="binary"):
        validate_water_mask(np.array([[0, 2]]))


def test_mask_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        validate_water_mask(np.array([[0.0, np.nan]]))


def test_segment_water_hsv_returns_binary_mask():
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    mask = segment_water_hsv(image)

    assert mask.shape == (20, 30)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_segment_water_hsv_rejects_grayscale():
    image = np.zeros((20, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        segment_water_hsv(image)
