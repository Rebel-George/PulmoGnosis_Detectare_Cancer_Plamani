"""Spatial operations: resampling, CT-PET alignment, extracting the model's input cube."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def resample_isotropic(
    image: sitk.Image,
    spacing_mm: float = 1.0,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample to cubic voxels of `spacing_mm`, keeping origin and direction."""
    old_spacing = np.array(image.GetSpacing())
    old_size = np.array(image.GetSize())
    new_size = np.maximum(np.round(old_size * old_spacing / spacing_mm), 1).astype(int)
    return sitk.Resample(
        image,
        [int(s) for s in new_size],
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        (spacing_mm,) * 3,
        image.GetDirection(),
        default_value,
        image.GetPixelID(),
    )


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample `moving` onto the grid of `reference` using physical coordinates.

    PET and CT from the same PET/CT exam share a frame of reference, so an
    identity transform in physical space aligns them; this is the baseline
    co-registration step.
    """
    return sitk.Resample(moving, reference, sitk.Transform(), interpolator, default_value, moving.GetPixelID())


def resample_around(
    image: sitk.Image,
    center_mm: tuple[float, float, float],
    size: int,
    spacing_mm: float,
    direction: tuple[float, ...] | None = None,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> np.ndarray:
    """Cube of `size`³ voxels of `spacing_mm`, centred on a physical point, as a (z, y, x) array.

    Works directly from the image's native grid, so CT and PET are each
    interpolated only once. Regions outside the image get `default_value`.
    """
    direction = direction or image.GetDirection()
    axes = np.array(direction, dtype=float).reshape(3, 3)
    half_extent = (size - 1) / 2 * spacing_mm
    origin = np.array(center_mm) - axes @ np.full(3, half_extent)

    reference = sitk.Image([size] * 3, image.GetPixelID())
    reference.SetSpacing((spacing_mm,) * 3)
    reference.SetDirection(direction)
    reference.SetOrigin(tuple(origin))
    return to_array(resample_to_reference(image, reference, interpolator, default_value))


def to_array(image: sitk.Image) -> np.ndarray:
    """SimpleITK image as a numpy array indexed (z, y, x)."""
    return sitk.GetArrayFromImage(image)
