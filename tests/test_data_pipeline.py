"""IDC selection, SEG reading and cube building — on fake index tables and the synthetic phantom."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulmognosis.data.dicom import scan_folder
from pulmognosis.data.idc import select_nsclc_radiogenomics, size_report
from pulmognosis.data.seg import find_seg_files, read_seg
from pulmognosis.data.synthetic import TEST, TUMOUR_CENTER_MM, TUMOUR_RADIUS_MM, write_demo_case

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_idc_selection_keeps_needed_series_for_seg_patients_only():
    rows = [
        # R01-001 has a SEG: keep both CTs, the SEG and the corrected PET, drop the uncorrected PET
        ("R01-001", "CT", "THORAX 1.0 B45f", 150), ("R01-001", "SEG", "3D Slicer segmentation result", 1),
        ("R01-001", "CT", "CT Images", 60), ("R01-001", "PT", "PET_BODY_CTAC", 30), ("R01-001", "PT", "PET_BODY_NO_AC", 30),
        ("R01-002", "CT", "THORAX", 150),            # no SEG → not selected
        ("AMC-003", "CT", "CT", 150), ("AMC-003", "SEG", "seg", 1),  # not an R01 patient
    ]
    index = pd.DataFrame(rows, columns=["PatientID", "Modality", "SeriesDescription", "series_size_MB"])
    index["collection_id"] = "nsclc_radiogenomics"
    index["SeriesInstanceUID"] = [f"uid{i}" for i in range(len(index))]

    sel = select_nsclc_radiogenomics(index)
    assert set(sel["PatientID"]) == {"R01-001"}
    assert sorted(sel["SeriesDescription"]) == sorted(["THORAX 1.0 B45f", "3D Slicer segmentation result", "CT Images", "PET_BODY_CTAC"])
    assert size_report(sel).loc["TOTAL", "series"] == 4
    assert select_nsclc_radiogenomics(index, patients=["R01-002"]).empty


def test_seg_reader_finds_tumour_centre_and_ct(tmp_path):
    root = write_demo_case(tmp_path, TEST)
    segs = find_seg_files(root)
    assert len(segs) == 1
    mask = read_seg(segs[0])
    assert np.allclose(mask.center_mm, TUMOUR_CENTER_MM, atol=1.0)
    assert np.all(np.abs(mask.extent_mm - 2 * TUMOUR_RADIUS_MM) < 4)
    ct = next(s for s in scan_folder(root) if s.modality == "CT")
    assert mask.referenced_series_uid == ct.uid
    # the SEG is a single multi-frame file and must not be picked up as an image series
    assert {s.modality for s in scan_folder(root)} == {"CT", "PT"}


def test_build_cubes_writes_tumour_centred_cube(tmp_path):
    build = _load_script("build_cubes")
    patient = write_demo_case(tmp_path / "raw" / "DEMO-001", TEST)
    info = build.build_one(patient, tmp_path / "cubes" / "DEMO-001.npz")
    assert info["ct_choice"] == "referenced by SEG"
    with np.load(tmp_path / "cubes" / "DEMO-001.npz") as z:
        assert z["ct"].shape == (128, 128, 128) and z["ct"].dtype == np.float16
        assert -100 < float(z["ct"][64, 64, 64]) < 150  # tumour tissue at the centre
        assert np.allclose(z["center_mm"], TUMOUR_CENTER_MM, atol=1.0)


def test_build_cubes_reports_patients_without_seg(tmp_path):
    build = _load_script("build_cubes")
    write_demo_case(tmp_path / "no_seg", TEST, with_seg=False)
    with pytest.raises(ValueError, match="no SEG"):
        build.build_one(tmp_path / "no_seg", tmp_path / "x.npz")
