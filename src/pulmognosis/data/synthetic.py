"""Synthetic CT + PET phantom written as real DICOM files.

Used by the tests and by the app's demo button, so the full flow can be
exercised without downloading patient data. The phantom is a simplified
chest: body, two lungs, spine, heart, and one tumour in the right lung.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, PositronEmissionTomographyImageStorage, generate_uid

PATIENT_WEIGHT_KG = 70.0
INJECTED_DOSE_BQ = 300e6
F18_HALF_LIFE_S = 6586.2
INJECTION_TIME = "100000"  # 10:00:00
SCAN_TIME = "110000"  # 11:00:00, i.e. 3600 s after injection
TUMOUR_SUV = 8.0

# Tumour centre in physical coordinates (mm) and radius.
TUMOUR_CENTER_MM = (-60.0, 10.0, 0.0)  # x (patient left is +x), y, z
TUMOUR_RADIUS_MM = 14.0


def _grid(shape_zyx, spacing_zyx):
    """Physical coordinates (mm) of voxel centres for a grid centred on the origin."""
    axes = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape_zyx, spacing_zyx)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    return x, y, z


def _ellipsoid(x, y, z, center, radii):
    return sum(((c - c0) / r) ** 2 for c, c0, r in zip((x, y, z), center, radii)) <= 1.0


def _phantom_labels(shape_zyx, spacing_zyx):
    x, y, z = _grid(shape_zyx, spacing_zyx)
    return {
        "body": _ellipsoid(x, y, z, (0, 0, 0), (160, 110, 1e6)),
        "lung_r": _ellipsoid(x, y, z, (-70, -5, 0), (55, 75, 1e6)),
        "lung_l": _ellipsoid(x, y, z, (70, -5, 0), (55, 75, 1e6)),
        "heart": _ellipsoid(x, y, z, (15, 15, 0), (40, 35, 45)),
        "spine": _ellipsoid(x, y, z, (0, 80, 0), (15, 15, 1e6)),
        "tumour": _ellipsoid(x, y, z, TUMOUR_CENTER_MM, (TUMOUR_RADIUS_MM,) * 3),
    }


def make_ct_hu(shape_zyx=(60, 160, 160), spacing_zyx=(2.5, 2.0, 2.0), seed=0) -> np.ndarray:
    m = _phantom_labels(shape_zyx, spacing_zyx)
    hu = np.full(shape_zyx, -1000.0)
    hu[m["body"]] = 40.0
    hu[m["lung_r"] | m["lung_l"]] = -850.0
    hu[m["heart"]] = 50.0
    hu[m["spine"]] = 700.0
    hu[m["tumour"]] = 30.0
    return hu + np.random.default_rng(seed).normal(0, 15, shape_zyx)


def make_pet_suv(shape_zyx=(30, 80, 80), spacing_zyx=(5.0, 4.0, 4.0)) -> np.ndarray:
    m = _phantom_labels(shape_zyx, spacing_zyx)
    suv = np.zeros(shape_zyx)
    suv[m["body"]] = 1.0
    suv[m["lung_r"] | m["lung_l"]] = 0.4
    suv[m["heart"]] = 2.5
    suv[m["tumour"]] = TUMOUR_SUV
    return suv


def _decayed_dose() -> float:
    elapsed = 3600.0
    return INJECTED_DOSE_BQ * 2 ** (-elapsed / F18_HALF_LIFE_S)


def _write_series(out_dir: Path, volume: np.ndarray, spacing_zyx, modality: str, common: dict, slope: float, intercept: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    sop_class = CTImageStorage if modality == "CT" else PositronEmissionTomographyImageStorage
    series_uid = generate_uid()
    nz, ny, nx = volume.shape
    sz, sy, sx = spacing_zyx
    origin = [-(nx - 1) / 2 * sx, -(ny - 1) / 2 * sy, -(nz - 1) / 2 * sz]
    stored = np.round((volume - intercept) / slope).astype(np.int16)

    for k in range(nz):
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
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + k * sz]
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
        ds.PixelData = stored[k].tobytes()
        ds.save_as(out_dir / f"{modality}_{k + 1:04d}.dcm", enforce_file_format=True)


def write_demo_case(root: str | Path) -> Path:
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
    ct_spacing = (2.5, 2.0, 2.0)
    _write_series(root / "CT", make_ct_hu(spacing_zyx=ct_spacing), ct_spacing, "CT", common, slope=1.0, intercept=-1024.0)

    pet_spacing = (5.0, 4.0, 4.0)
    activity = make_pet_suv(spacing_zyx=pet_spacing) * _decayed_dose() / (PATIENT_WEIGHT_KG * 1000.0)
    slope = float(activity.max()) / 30000.0
    _write_series(root / "PET", activity, pet_spacing, "PT", common, slope=slope, intercept=0.0)
    return root
