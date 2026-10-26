"""Synthetic CT + PET phantom written as real DICOM files.

Used by the tests and by the app's demo button, so the full flow can be
exercised without downloading patient data. The phantom is a simplified
chest: body, two lungs, spine, heart, and one tumour in the right lung.

`DEMO` matches a typical chest PET/CT (512×512 CT at ~0.7 mm, 1.25 mm slices;
4 mm PET); `TEST` is a small, fast version with the same anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, PositronEmissionTomographyImageStorage, generate_uid

PATIENT_WEIGHT_KG = 70.0
INJECTED_DOSE_BQ = 300e6
F18_HALF_LIFE_S = 6586.2
INJECTION_TIME = "100000"  # 10:00:00
SCAN_TIME = "110000"  # 11:00:00, i.e. 3600 s after injection
TUMOUR_SUV = 8.0

# Tumour centre in physical coordinates (mm, DICOM LPS: +x is the patient's left) and radius.
TUMOUR_CENTER_MM = (-60.0, 10.0, 0.0)
TUMOUR_RADIUS_MM = 14.0


@dataclass(frozen=True)
class PhantomSpec:
    ct_shape_zyx: tuple[int, int, int]
    ct_spacing_zyx: tuple[float, float, float]
    pet_shape_zyx: tuple[int, int, int]
    pet_spacing_zyx: tuple[float, float, float]


DEMO = PhantomSpec((120, 512, 512), (1.25, 0.7, 0.7), (50, 128, 128), (3.0, 4.0, 4.0))
TEST = PhantomSpec((60, 160, 160), (2.5, 2.0, 2.0), (30, 80, 80), (5.0, 4.0, 4.0))


def _origin(shape_zyx, spacing_zyx) -> tuple[float, float, float]:
    """(x, y, z) of the first voxel, for a grid centred on the physical origin."""
    return tuple(-(n - 1) / 2 * s for n, s in zip(shape_zyx[::-1], spacing_zyx[::-1]))


def _slice_labels(shape_zyx, spacing_zyx, k: int) -> dict[str, np.ndarray]:
    """Boolean masks of each structure on axial slice `k`."""
    _, ny, nx = shape_zyx
    sz, sy, sx = spacing_zyx
    ox, oy, oz = _origin(shape_zyx, spacing_zyx)
    z = oz + k * sz
    y, x = np.meshgrid(oy + np.arange(ny) * sy, ox + np.arange(nx) * sx, indexing="ij")

    def ellipse(cx, cy, rx, ry):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        dz = ((z - cz) / rz) ** 2
        return ellipse(cx, cy, rx * np.sqrt(max(1 - dz, 0)), ry * np.sqrt(max(1 - dz, 0))) if dz < 1 else np.zeros_like(x, bool)

    tx, ty, tz = TUMOUR_CENTER_MM
    return {
        "body": ellipse(0, 0, 160, 110),
        "lungs": ellipse(-70, -5, 55, 75) | ellipse(70, -5, 55, 75),
        "heart": ellipsoid(15, 15, 0, 40, 35, 45),
        "spine": ellipse(0, 80, 15, 15),
        "tumour": ellipsoid(tx, ty, tz, TUMOUR_RADIUS_MM, TUMOUR_RADIUS_MM, TUMOUR_RADIUS_MM),
    }


def ct_slice_hu(spec: PhantomSpec, k: int, rng: np.random.Generator) -> np.ndarray:
    m = _slice_labels(spec.ct_shape_zyx, spec.ct_spacing_zyx, k)
    hu = np.full(m["body"].shape, -1000.0)
    hu[m["body"]] = 40.0
    hu[m["lungs"]] = -850.0
    hu[m["heart"]] = 50.0
    hu[m["spine"]] = 700.0
    hu[m["tumour"]] = 30.0
    return hu + rng.normal(0, 15, hu.shape)


def pet_slice_suv(spec: PhantomSpec, k: int) -> np.ndarray:
    m = _slice_labels(spec.pet_shape_zyx, spec.pet_spacing_zyx, k)
    suv = np.zeros(m["body"].shape)
    suv[m["body"]] = 1.0
    suv[m["lungs"]] = 0.4
    suv[m["heart"]] = 2.5
    suv[m["tumour"]] = TUMOUR_SUV
    return suv


def _decayed_dose() -> float:
    elapsed = 3600.0
    return INJECTED_DOSE_BQ * 2 ** (-elapsed / F18_HALF_LIFE_S)


def _write_series(
    out_dir: Path,
    shape_zyx,
    spacing_zyx,
    slice_values: Callable[[int], np.ndarray],
    modality: str,
    common: dict,
    slope: float,
    intercept: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sop_class = CTImageStorage if modality == "CT" else PositronEmissionTomographyImageStorage
    series_uid = generate_uid()
    nz, ny, nx = shape_zyx
    sz, sy, sx = spacing_zyx
    ox, oy, oz = _origin(shape_zyx, spacing_zyx)

    for k in range(nz):
        stored = np.round((slice_values(k) - intercept) / slope).astype(np.int16)
        sop_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = sop_class
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        for key, value in common.items():
            setattr(ds, key, value)
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = sop_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = modality
        ds.SeriesDescription = f"Fantomă sintetică {modality}"
        ds.SeriesNumber = 1 if modality == "CT" else 2
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [ox, oy, oz + k * sz]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [sy, sx]
        ds.SliceThickness = sz
        ds.Rows, ds.Columns = ny, nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = f"{slope:.6g}"
        ds.RescaleIntercept = f"{intercept:.6g}"
        if modality == "PT":
            ds.Units = "BQML"
            ds.DecayCorrection = "START"
            ds.SeriesTime = SCAN_TIME
            radio = Dataset()
            radio.RadionuclideTotalDose = f"{INJECTED_DOSE_BQ:.6g}"
            radio.RadionuclideHalfLife = f"{F18_HALF_LIFE_S}"
            radio.RadiopharmaceuticalStartTime = INJECTION_TIME
            ds.RadiopharmaceuticalInformationSequence = [radio]
        ds.PixelData = stored.tobytes()
        ds.save_as(out_dir / f"{modality}_{k + 1:04d}.dcm", enforce_file_format=True)


def write_demo_case(root: str | Path, spec: PhantomSpec = DEMO, seed: int = 0) -> Path:
    """Write the phantom's CT and PET series under `root` and return it."""
    root = Path(root)
    common = {
        "SpecificCharacterSet": "ISO_IR 192",  # UTF-8, for Romanian diacritics
        "PatientName": "Demo^Fantoma",
        "PatientID": "DEMO-001",
        "PatientSex": "M",
        "PatientAge": "064Y",
        "PatientWeight": f"{PATIENT_WEIGHT_KG}",
        "StudyInstanceUID": generate_uid(),
        "FrameOfReferenceUID": generate_uid(),
        "StudyDescription": "Fantomă sintetică PulmoGnosis",
    }
    rng = np.random.default_rng(seed)
    _write_series(root / "CT", spec.ct_shape_zyx, spec.ct_spacing_zyx, lambda k: ct_slice_hu(spec, k, rng),
                  "CT", common, slope=1.0, intercept=-1024.0)

    bq_per_suv = _decayed_dose() / (PATIENT_WEIGHT_KG * 1000.0)
    slope = TUMOUR_SUV * bq_per_suv / 30000.0
    _write_series(root / "PET", spec.pet_shape_zyx, spec.pet_spacing_zyx, lambda k: pet_slice_suv(spec, k) * bq_per_suv,
                  "PT", common, slope=slope, intercept=0.0)
    return root
