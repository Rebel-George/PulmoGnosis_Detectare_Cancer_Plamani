"""Stage 0: turn raw DICOM series into an aligned, analysis-ready case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from pulmognosis.data.dicom import SeriesInfo, load_series, read_header
from pulmognosis.preprocessing.intensity import suv_scale_factor
from pulmognosis.preprocessing.spatial import resample_around, resample_to_reference

SMOKING_CHOICES = ("niciodată", "fost fumător", "fumător actual")
SEX_CHOICES = ("masculin", "feminin")

# The classifier's input: a cube of MODEL_PATCH_SIZE³ voxels of MODEL_SPACING_MM,
# i.e. 96 mm per side, close to the in-plane resolution of a typical chest CT.
MODEL_PATCH_SIZE = 128
MODEL_SPACING_MM = 0.75
MODEL_PATCH_MM = MODEL_PATCH_SIZE * MODEL_SPACING_MM


@dataclass
class ClinicalData:
    age: int
    sex: str
    smoking: str


@dataclass
class Case:
    """A patient after Stage 0.

    - `ct`: CT in HU at the scan's native resolution.
    - `pet`: PET in SUV resampled onto the CT grid (for display).
    - `pet_native`: PET in SUV on its own grid (for localisation and patches).
    """

    ct: sitk.Image
    pet: sitk.Image | None
    pet_native: sitk.Image | None = None
    pet_warning: str | None = None

    @property
    def has_pet(self) -> bool:
        return self.pet is not None

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        return tuple(self.ct.GetSpacing())

    def index_to_mm(self, index_zyx) -> tuple[float, float, float]:
        z, y, x = (int(i) for i in index_zyx)
        return self.ct.TransformIndexToPhysicalPoint((x, y, z))

    def mm_to_index(self, point_mm) -> tuple[int, int, int]:
        """Nearest CT voxel (z, y, x) to a physical point, clipped to the volume."""
        x, y, z = self.ct.TransformPhysicalPointToIndex(tuple(float(c) for c in point_mm))
        size = self.ct.GetSize()
        clip = lambda v, n: int(min(max(v, 0), n - 1))
        return clip(z, size[2]), clip(y, size[1]), clip(x, size[0])


def prepare_case(ct_info: SeriesInfo, pet_info: SeriesInfo | None) -> Case:
    """Load CT (and PET in SUV), keeping the CT at native resolution and aligning PET onto it.

    If the PET header lacks the tags needed for SUV, the PET is kept in its
    stored units and a warning is returned instead of failing the whole case.
    """
    ct = load_series(ct_info)
    if pet_info is None:
        return Case(ct=ct, pet=None)

    pet_native = load_series(pet_info)
    warning = None
    try:
        pet_native = pet_native * suv_scale_factor(read_header(pet_info))
    except ValueError as exc:
        warning = f"PET-ul nu a putut fi convertit în SUV ({exc}); se afișează valorile brute."
    pet = resample_to_reference(pet_native, ct, sitk.sitkLinear, default_value=0.0)
    return Case(ct=ct, pet=pet, pet_native=pet_native, pet_warning=warning)


def extract_model_patch(case: Case, center_zyx) -> tuple[np.ndarray, np.ndarray | None]:
    """The classifier's input cube around a CT voxel: CT (HU) and PET (SUV), each (z, y, x).

    Both are resampled from their native grids to MODEL_SPACING_MM, aligned
    to the CT's orientation; outside the scan, CT is air (-1000) and PET is 0.
    """
    center_mm = case.index_to_mm(center_zyx)
    direction = case.ct.GetDirection()
    ct = resample_around(case.ct, center_mm, MODEL_PATCH_SIZE, MODEL_SPACING_MM, direction, default_value=-1000.0)
    pet = None
    if case.pet_native is not None:
        pet = resample_around(case.pet_native, center_mm, MODEL_PATCH_SIZE, MODEL_SPACING_MM, direction, default_value=0.0)
    return ct, pet
