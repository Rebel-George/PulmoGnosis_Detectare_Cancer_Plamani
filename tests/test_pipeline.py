"""End-to-end Stage 0 on the synthetic phantom: DICOM on disk → aligned CT (HU) + PET (SUV)."""

import numpy as np
import pytest

from pulmognosis.data.dicom import pick_series, scan_folder
from pulmognosis.data.synthetic import TUMOUR_CENTER_MM, TUMOUR_SUV, write_demo_case
from pulmognosis.inference import PlaceholderPredictor
from pulmognosis.pipeline import ClinicalData, prepare_case
from pulmognosis.preprocessing.spatial import to_array


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = write_demo_case(tmp_path_factory.mktemp("demo"))
    series = scan_folder(root)
    return prepare_case(pick_series(series, "CT"), pick_series(series, "PT"))


def test_scan_finds_ct_and_pet(tmp_path):
    series = scan_folder(write_demo_case(tmp_path))
    assert {s.modality: s.num_slices for s in series} == {"CT": 60, "PT": 30}


def test_ct_is_in_hounsfield_units_on_1mm_grid(case):
    ct = to_array(case.ct)
    assert case.ct.GetSpacing() == (1.0, 1.0, 1.0)
    assert np.median(ct[ct < -500]) == pytest.approx(-900, abs=150)  # air and lung
    assert ct.max() > 600  # spine


def test_pet_is_converted_to_suv_and_aligned_with_ct(case):
    assert case.pet_warning is None
    pet = to_array(case.pet)
    assert pet.shape == to_array(case.ct).shape
    assert pet.max() == pytest.approx(TUMOUR_SUV, rel=0.02)

    # The PET hotspot must sit where the phantom put the tumour (in mm).
    z, y, x = np.unravel_index(np.argmax(pet), pet.shape)
    hot_mm = case.ct.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    assert np.linalg.norm(np.subtract(hot_mm, TUMOUR_CENTER_MM)) < 15


def test_placeholder_localises_on_the_tumour_and_flags_itself(case):
    predictor = PlaceholderPredictor()
    location = predictor.localise(case)
    point_mm = case.ct.TransformIndexToPhysicalPoint(tuple(int(c) for c in location.center_zyx[::-1]))
    assert np.linalg.norm(np.subtract(point_mm, TUMOUR_CENTER_MM)) < 5

    prediction = predictor.predict(case, location, ClinicalData(60, "masculin", "fost fumător"))
    assert prediction.is_placeholder
    assert sum(prediction.histology.values()) == pytest.approx(1.0)
