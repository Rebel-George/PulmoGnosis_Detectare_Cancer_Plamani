"""DICOM SEG tumour masks (as in NSCLC-Radiogenomics): which CT they belong to and where the tumour is.

A SEG file is one multi-frame DICOM: each frame is a binary 2D mask on one
slice, with its own position in patient space. We only need the tumour's
centre and extent in millimetres, so the mask is never resampled onto the CT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class SegMask:
    path: Path
    patient_id: str
    study_uid: str
    referenced_series_uid: str | None  # the CT series the mask was drawn on
    center_mm: np.ndarray  # (x, y, z) centroid of the tumour voxels
    bbox_min_mm: np.ndarray
    bbox_max_mm: np.ndarray
    n_voxels: int

    @property
    def extent_mm(self) -> np.ndarray:
        return self.bbox_max_mm - self.bbox_min_mm


def find_seg_files(root: str | Path) -> list[Path]:
    """All DICOM files under `root` whose Modality is SEG."""
    found = []
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["Modality"])
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        if getattr(ds, "Modality", None) == "SEG":
            found.append(path)
    return sorted(found)


def _frame_geometry(ds: pydicom.Dataset, frame: pydicom.Dataset):
    """Position, row/column directions and pixel spacing of one frame (per-frame values win over shared)."""
    shared = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else pydicom.Dataset()

    def pick(name):
        if name in frame:
            return frame[name].value[0]
        if name in shared:
            return shared[name].value[0]
        raise ValueError(f"SEG frame has no {name}")

    position = np.array(pick("PlanePositionSequence").ImagePositionPatient, dtype=float)
    orientation = np.array(pick("PlaneOrientationSequence").ImageOrientationPatient, dtype=float)
    row_spacing, col_spacing = (float(v) for v in pick("PixelMeasuresSequence").PixelSpacing)
    return position, orientation[:3], orientation[3:], row_spacing, col_spacing


def read_seg(path: str | Path, segment_number: int | None = None) -> SegMask:
    """Centroid and bounding box (mm) of the mask; all segments unless `segment_number` is given."""
    path = Path(path)
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    if pixels.ndim == 2:
        pixels = pixels[None]

    total, count = np.zeros(3), 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for frame_pixels, frame in zip(pixels, ds.PerFrameFunctionalGroupsSequence):
        if segment_number is not None:
            seg_id = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            if int(seg_id) != segment_number:
                continue
        rows, cols = np.nonzero(frame_pixels)
        if rows.size == 0:
            continue
        position, row_dir, col_dir, row_spacing, col_spacing = _frame_geometry(ds, frame)
        # Column index moves along the row direction; row index along the column direction.
        points = position + np.outer(cols * col_spacing, row_dir) + np.outer(rows * row_spacing, col_dir)
        total += points.sum(axis=0)
        count += len(points)
        lo, hi = np.minimum(lo, points.min(axis=0)), np.maximum(hi, points.max(axis=0))

    if count == 0:
        raise ValueError(f"SEG {path} contains no tumour voxels")

    referenced = None
    if "ReferencedSeriesSequence" in ds:
        referenced = str(ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
    return SegMask(
        path=path,
        patient_id=str(getattr(ds, "PatientID", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        referenced_series_uid=referenced,
        center_mm=total / count,
        bbox_min_mm=lo,
        bbox_max_mm=hi,
        n_voxels=count,
    )
