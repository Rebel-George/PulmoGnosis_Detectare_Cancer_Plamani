"""Stage 0: turn raw DICOM series into an aligned, analysis-ready case."""

from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk

from pulmognosis.data.dicom import SeriesInfo, load_series, read_header
from pulmognosis.preprocessing.intensity import suv_scale_factor
from pulmognosis.preprocessing.spatial import resample_isotropic, resample_to_reference

SMOKING_CHOICES = ("niciodată", "fost fumător", "fumător actual")
SEX_CHOICES = ("masculin", "feminin")


@dataclass
class ClinicalData:
    age: int
    sex: str
    smoking: str


@dataclass
class Case:
    """A patient after Stage 0: CT in HU and PET in SUV on the same 1 mm grid."""

    ct: sitk.Image
    pet: sitk.Image | None
    pet_warning: str | None = None

    @property
    def has_pet(self) -> bool:
        return self.pet is not None


def prepare_case(ct_info: SeriesInfo, pet_info: SeriesInfo | None, spacing_mm: float = 1.0) -> Case:
    """Load CT (and PET), resample CT to isotropic spacing and align PET onto it.

    If the PET header lacks the tags needed for SUV, the PET is kept in its
    stored units and a warning is returned instead of failing the whole case.
    """
    ct = resample_isotropic(load_series(ct_info), spacing_mm, sitk.sitkLinear, default_value=-1000.0)
    if pet_info is None:
        return Case(ct=ct, pet=None)

    pet_raw = load_series(pet_info)
    warning = None
    try:
        pet_raw = pet_raw * suv_scale_factor(read_header(pet_info))
    except ValueError as exc:
        warning = f"PET-ul nu a putut fi convertit în SUV ({exc}); se afișează valorile brute."
    pet = resample_to_reference(pet_raw, ct, sitk.sitkLinear, default_value=0.0)
    return Case(ct=ct, pet=pet, pet_warning=warning)
