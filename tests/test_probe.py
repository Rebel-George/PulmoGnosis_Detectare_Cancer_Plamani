"""Linear probe (Model C, frozen features): input preparation, out-of-fold CV, feature extraction."""

import numpy as np
import pandas as pd
import torch

from pulmognosis.classification.dataset import labelled_patients
from pulmognosis.classification.model3d import FEATURES, ResNet3DClassifier
from pulmognosis.classification.probe import cross_validate_probe, extract_features, prepare_input
from pulmognosis.evaluation.metrics import binary_metrics


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
