import math

import numpy as np
import pytest
from pydicom.dataset import Dataset

from pulmognosis.preprocessing.intensity import _dicom_time_to_seconds, normalize_ct, suv_scale_factor, window


def test_window_maps_edges_to_0_and_1():
    hu = np.array([-2000.0, -1000.0, -300.0, 400.0, 3000.0])
    out = window(hu, center=-300.0, width=1400.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_normalize_ct_clips_to_model_range():
    np.testing.assert_allclose(normalize_ct(np.array([-1500.0, -1000.0, 400.0, 1000.0])), [0, 0, 1, 1])


@pytest.mark.parametrize("value,seconds", [("110000", 39600), ("101530.5", 36930.5), ("09", 32400)])
def test_dicom_time_parsing(value, seconds):
    assert _dicom_time_to_seconds(value) == pytest.approx(seconds)


def _pet_header(weight=70.0, dose=300e6, start="100000", series="110000"):
    ds = Dataset()
    ds.Units = "BQML"
    ds.PatientWeight = weight
    ds.SeriesTime = series
    radio = Dataset()
    radio.RadionuclideTotalDose = dose
    radio.RadionuclideHalfLife = 6586.2
    radio.RadiopharmaceuticalStartTime = start
    ds.RadiopharmaceuticalInformationSequence = [radio]
    return ds


def test_suv_factor_matches_formula():
    decayed = 300e6 * math.exp(-math.log(2) * 3600 / 6586.2)
    assert suv_scale_factor(_pet_header()) == pytest.approx(70_000 / decayed)


def test_suv_factor_handles_scan_after_midnight():
    # Injected 23:30, scanned 00:30 → 3600 s elapsed, same as the regular case.
    assert suv_scale_factor(_pet_header(start="233000", series="003000")) == pytest.approx(suv_scale_factor(_pet_header()))


def test_suv_factor_prefers_philips_private_tag():
    ds = _pet_header()
    ds.add_new((0x7053, 0x1000), "DS", "0.00042")
    assert suv_scale_factor(ds) == pytest.approx(0.00042)


def test_suv_factor_requires_weight():
    with pytest.raises(ValueError, match="PatientWeight"):
        suv_scale_factor(_pet_header(weight=0))


def test_suv_factor_rejects_non_bqml_units():
    ds = _pet_header()
    ds.Units = "CNTS"
    with pytest.raises(ValueError, match="units"):
        suv_scale_factor(ds)
