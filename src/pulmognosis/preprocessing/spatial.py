"""Spatial operations: resampling, CT-PET alignment, patch cropping."""

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


def to_array(image: sitk.Image) -> np.ndarray:
    """SimpleITK image as a numpy array indexed (z, y, x)."""
    return sitk.GetArrayFromImage(image)


def crop_patch(volume: np.ndarray, center_zyx: tuple[int, int, int], size: int = 64, fill: float = 0.0) -> np.ndarray:
    """Cube of side `size` centred on `center_zyx`, padded with `fill` at the borders."""
    half = size // 2
    patch = np.full((size, size, size), fill, dtype=volume.dtype)
    src, dst = [], []
    for c, dim in zip(center_zyx, volume.shape):
        start = int(c) - half
        lo, hi = max(start, 0), min(start + size, dim)
        src.append(slice(lo, hi))
        dst.append(slice(lo - start, hi - start))
    patch[tuple(dst)] = volume[tuple(src)]
    return patch
