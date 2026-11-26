"""Linear probe (Model C, frozen features): input preparation, out-of-fold CV, feature extraction."""

import numpy as np
import pandas as pd
import pytest
import torch

from pulmognosis.classification.dataset import labelled_patients
from pulmognosis.classification.model3d import FEATURES, ResNet3DClassifier
from pulmognosis.classification.probe import (ProbeModel, _pipeline, cross_validate_probe, extract_features, fit_final,
                                              prepare_input)
from pulmognosis.evaluation.metrics import binary_metrics
from pulmognosis.inference import ClinicalPredictor

FEATURES_N = FEATURES  # 512 values per cube


def test_prepare_input_crops_the_centre_and_standardises():
    cube = np.zeros((1, 8, 8, 8), dtype=np.float32)
    cube[0, 2:6, 2:6, 2:6] = 1.0  # bright centre
    out = prepare_input(cube, "zscore", crop=4)
    assert out.shape == (1, 4, 4, 4)
    assert np.allclose(out, 0.0)  # the 4³ centre is uniform, so it standardises to zeros
    full = prepare_input(cube, "zscore")
    assert full.shape == (1, 8, 8, 8)
    assert abs(full.mean()) < 1e-5 and abs(full.std() - 1) < 1e-4
    assert np.array_equal(prepare_input(cube, "unit"), cube)


def test_probe_cv_is_out_of_fold_and_finds_a_planted_signal():
    rng = np.random.default_rng(0)
    n = 120
    y = (np.arange(n) % 4 == 0).astype(int)  # 25 % positive, like squamous vs. adeno
    x = rng.normal(size=(n, 16))
    x[:, 0] += 2.0 * y  # one informative feature among noise (pooled AUC 0.80–0.92 over seeds 0–4)
    folds = np.arange(n) % 5
    oof = cross_validate_probe(x, y, folds)
    assert sorted(oof["index"]) == list(range(n))  # every patient predicted exactly once
    assert all(oof.loc[oof["fold"] == f, "index"].map(lambda i: folds[i] == f).all() for f in range(5))
    assert (oof["threshold"] == 0.5).all()
    assert binary_metrics(oof["y"], oof["prob"])["auc"] > 0.75


def test_feature_extraction_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(3):
        np.savez(tmp_path / f"P{i}.npz", ct=rng.normal(-500, 200, (32, 32, 32)).astype(np.float16))
        rows.append({"patient_id": f"P{i}", "histology": ["adenocarcinoma", "squamous", "nos"][i]})
    df = labelled_patients(pd.DataFrame(rows), tmp_path, "histology")
    assert list(df["patient_id"]) == ["P0", "P1"]  # NOS has no adeno/squamous label
    encoder = ResNet3DClassifier(in_channels=1, pretrained="none").encoder
    feats = extract_features(list(df["path"]), encoder, torch.device("cpu"), "zscore", crop=16)
    assert feats.shape == (2, FEATURES) and np.isfinite(feats).all()


def test_exported_probe_matches_the_fitted_pipeline(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 16))
    y = (rng.random(80) < 0.25).astype(int)
    x[:, 0] += 1.5 * y
    model = fit_final(x, y, c=1.0, target="histology", classes=["adenocarcinom", "carcinom scuamos"],
                      pretrained="none", input_norm="zscore", crop=0, cv_auc=[0.6, 0.5, 0.7])
    loaded = ProbeModel.load(model.save(tmp_path / "probe.json"))
    reference = _pipeline(1.0).fit(x, y).predict_proba(x)[:, 1]
    assert np.allclose(loaded.predict_proba(x), reference, atol=1e-9)
    assert loaded.n_patients == 80 and loaded.n_positive == int(y.sum())


def test_image_predictor_scores_histology_on_the_demo_case(tmp_path):
    from pulmognosis.classification.clinical import FEATURES as CLINICAL_FEATURES
    from pulmognosis.classification.clinical import ClinicalModel
    from pulmognosis.data.dicom import pick_series, scan_folder
    from pulmognosis.data.synthetic import TEST, write_demo_case
    from pulmognosis.inference import HistologyProbe, ImagePredictor, Localisation
    from pulmognosis.pipeline import ClinicalData, prepare_case

    series = scan_folder(write_demo_case(tmp_path / "case", TEST))
    case = prepare_case(pick_series(series, "CT"), pick_series(series, "PT"))
    weights = {t: {"intercept": -1.0, **{f: 0.1 for f in CLINICAL_FEATURES}} for t in ("egfr", "kras")}
    clinical = ClinicalModel(60.0, 10.0, weights, {"egfr": 172, "kras": 171}, {"egfr": 43, "kras": 38})
    rng = np.random.default_rng(3)
    probe = ProbeModel(target="histology", classes=["adenocarcinom", "carcinom scuamos"], pretrained="none",
                       input_norm="zscore", crop=0, mean=[0.0] * FEATURES_N, scale=[1.0] * FEATURES_N,
                       coef=rng.normal(0, 0.01, FEATURES_N).tolist(), intercept=0.0, c=1.0,
                       n_patients=141, n_positive=29, cv_auc=[0.64, 0.53, 0.74])

    pred = ImagePredictor(clinical, HistologyProbe(probe)).predict(
        case, Localisation((30, 80, 80), "test"), ClinicalData(70, "feminin", "niciodată"))
    assert set(pred.histology) == {"adenocarcinom", "carcinom scuamos"}
    assert sum(pred.histology.values()) == pytest.approx(1.0)
    assert 0.0 < pred.histology["carcinom scuamos"] < 1.0
    assert pred.placeholder_outputs == frozenset() and not pred.is_placeholder
    assert "model de imagine C" in pred.sources["histology"] and "0.64" in pred.sources["histology"]
    assert pred.egfr == pytest.approx(ClinicalPredictor(clinical).predict(
        case, Localisation((30, 80, 80), "test"), ClinicalData(70, "feminin", "niciodată")).egfr)
