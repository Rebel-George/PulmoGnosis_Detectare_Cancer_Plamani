from pathlib import Path

from pulmognosis.data.dicom import SeriesInfo, is_uncorrected_pet, pick_series


def _series(modality, description, n):
    return SeriesInfo(uid=description, modality=modality, description=description, files=[Path(f"{i}.dcm") for i in range(n)])


def test_pick_series_prefers_attenuation_corrected_pet():
    # Real NSCLC-Radiogenomics layout: the uncorrected PET has one more slice.
    corrected = _series("PT", "PET_BODY_CTAC", 204)
    uncorrected = _series("PT", "PET_BODY_NO_AC", 205)
    assert is_uncorrected_pet(uncorrected) and not is_uncorrected_pet(corrected)
    assert pick_series([uncorrected, corrected], "PT") is corrected


def test_pick_series_takes_largest_ct_and_handles_missing_modality():
    small, large = _series("CT", "scout", 3), _series("CT", "THORAX 1.0 B45f", 304)
    assert pick_series([small, large], "CT") is large
    assert pick_series([small, large], "PT") is None
