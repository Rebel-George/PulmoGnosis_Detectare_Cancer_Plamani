"""Discover and load DICOM series (CT, PET) into SimpleITK images."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


@dataclass
class SeriesInfo:
    uid: str
    modality: str
    description: str
    files: list[Path] = field(default_factory=list)

    @property
    def num_slices(self) -> int:
        return len(self.files)


def scan_folder(root: str | Path) -> list[SeriesInfo]:
    """Group every readable DICOM file under `root` by SeriesInstanceUID.

    Series with fewer than 2 slices (scout images, dose reports) are dropped.
    """
    groups: dict[str, SeriesInfo] = {}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None or "ImagePositionPatient" not in ds:
            continue
        if uid not in groups:
            groups[uid] = SeriesInfo(
                uid=uid,
                modality=str(getattr(ds, "Modality", "")),
                description=str(getattr(ds, "SeriesDescription", "")),
            )
        groups[uid].files.append(path)
    return sorted(
        (s for s in groups.values() if s.num_slices >= 2),
        key=lambda s: (s.modality, -s.num_slices),
    )


def is_uncorrected_pet(info: SeriesInfo) -> bool:
    """PET reconstructed without attenuation correction (e.g. 'PET_BODY_NO_AC'): not usable for SUV."""
    desc = info.description.upper().replace("-", "_").replace(" ", "_")
    return info.modality == "PT" and any(tag in desc for tag in ("NO_AC", "NOAC", "NAC", "NON_AC", "UNCORRECTED"))


def pick_series(series: list[SeriesInfo], modality: str) -> SeriesInfo | None:
    """Return the series of the given modality with the most slices.

    For PET, attenuation-corrected series are preferred over uncorrected ones,
    even when the uncorrected series has more slices.
    """
    candidates = [s for s in series if s.modality == modality]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (not is_uncorrected_pet(s), s.num_slices))


def _sorted_by_position(files: list[Path]) -> list[Path]:
    """Sort slice files along the slice normal (robust to unordered file names)."""
    keyed = defaultdict(list)
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        row, col = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
        normal = np.cross(row, col)
        keyed[float(np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float)))].append(f)
    return [f for pos in sorted(keyed) for f in keyed[pos]]


def load_series(info: SeriesInfo) -> sitk.Image:
    """Load a series as a float32 SimpleITK image.

    SimpleITK applies RescaleSlope / RescaleIntercept, so CT comes out in
    Hounsfield units and PET in the stored activity units (usually Bq/ml).
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in _sorted_by_position(info.files)])
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


def read_header(info: SeriesInfo) -> pydicom.Dataset:
    """Header of the first slice, for tags SimpleITK does not expose (e.g. PET dose)."""
    return pydicom.dcmread(info.files[0], stop_before_pixels=True)
