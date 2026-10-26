"""Intensity conversions: CT windowing and PET SUV."""

from __future__ import annotations

import math

import numpy as np
import pydicom

# Standard CT display windows as (center, width) in HU.
WINDOWS = {
    "lung": (-600.0, 1500.0),
    "mediastinum": (40.0, 400.0),
}

# Range used to normalise CT for the models.
CT_MODEL_RANGE = (-1000.0, 400.0)

# Philips private tag holding a ready-made SUV scale factor (used by some
# NSCLC-Radiogenomics PET series).
_PHILIPS_SUV_FACTOR = (0x7053, 0x1000)


def window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map HU to [0, 1] for display using a center/width window."""
    lo, hi = center - width / 2, center + width / 2
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def normalize_ct(hu: np.ndarray, lo: float = CT_MODEL_RANGE[0], hi: float = CT_MODEL_RANGE[1]) -> np.ndarray:
    """Clip CT to [lo, hi] HU and scale to [0, 1]."""
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def _dicom_time_to_seconds(value: str) -> float:
    """Parse a DICOM TM value (HHMMSS.ffffff, possibly truncated) into seconds."""
    value = str(value).strip()
    hours = int(value[0:2])
    minutes = int(value[2:4]) if len(value) >= 4 else 0
    seconds = float(value[4:]) if len(value) > 4 else 0.0
    return hours * 3600 + minutes * 60 + seconds


def suv_scale_factor(ds: pydicom.Dataset) -> float:
    """Factor that converts PET activity (Bq/ml) to body-weight SUV.

    SUV = activity * weight_g / injected_dose_at_scan_time_Bq, with the dose
    decay-corrected from injection to series start (the usual convention
    when DecayCorrection is START).

    Raises ValueError when the header lacks what the formula needs.
    """
    if _PHILIPS_SUV_FACTOR in ds:
        return float(ds[_PHILIPS_SUV_FACTOR].value)

    units = str(getattr(ds, "Units", "BQML")).upper()
    if units != "BQML":
        raise ValueError(f"Unsupported PET units {units!r}; expected BQML")

    weight_kg = float(getattr(ds, "PatientWeight", 0) or 0)
    if weight_kg <= 0:
        raise ValueError("PatientWeight missing; cannot compute SUV")

    try:
        radio = ds.RadiopharmaceuticalInformationSequence[0]
        dose_bq = float(radio.RadionuclideTotalDose)
        half_life_s = float(radio.RadionuclideHalfLife)
        start_s = _dicom_time_to_seconds(radio.RadiopharmaceuticalStartTime)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Radiopharmaceutical information missing; cannot compute SUV") from exc

    scan_s = _dicom_time_to_seconds(getattr(ds, "SeriesTime", None) or ds.AcquisitionTime)
    elapsed_s = (scan_s - start_s) % 86400  # tolerate scans crossing midnight
    decayed_dose_bq = dose_bq * math.exp(-math.log(2) * elapsed_s / half_life_s)
    return weight_kg * 1000.0 / decayed_dose_bq
