"""Cohort parsing, shared folds, metrics and Model A — on small synthetic tables, no real data needed."""

import numpy as np
import pandas as pd
import pytest

from pulmognosis.classification.clinical import ClinicalModel, cross_validate, fit_final
from pulmognosis.data.cohort import labels_from_clinical, modalities_from_digest
from pulmognosis.evaluation.metrics import binary_metrics, bootstrap_ci
from pulmognosis.evaluation.splits import fold_summary, make_splits
from pulmognosis.inference import ClinicalPredictor, Localisation
from pulmognosis.pipeline import ClinicalData


def _cohort(n=120, seed=0):
    """Synthetic cohort where never-smokers are much more often EGFR-mutant, as in the real data."""
    rng = np.random.default_rng(seed)
    smoking = rng.choice(["never", "former", "current"], n, p=[0.25, 0.6, 0.15])
    egfr = (rng.random(n) < np.where(smoking == "never", 0.6, 0.1)).astype(float)
    egfr[rng.random(n) < 0.15] = np.nan  # unknown labels
    kras = (rng.random(n) < 0.25).astype(float)
    return pd.DataFrame({
        "patient_id": [f"R01-{i:03d}" for i in range(n)],
        "egfr": egfr,
        "kras": kras,
        "age": rng.normal(68, 10, n),
        "sex": rng.choice(["M", "F"], n),
        "smoking": smoking,
    })


def test_labels_keep_unknown_as_missing_not_negative():
    clinical = pd.DataFrame({
        "Case ID": ["R01-001", "R01-002", "AMC-003", "R01-004"],
        "Histology ": ["Adenocarcinoma", "Squamous cell carcinoma", "Adenocarcinoma", "NSCLC NOS (not otherwise specified)"],
        "EGFR mutation status": ["Mutant", "Wildtype", "Unknown", "Not collected"],
        "KRAS mutation status": ["Wildtype", "Mutant", "Not collected", "Unknown"],
        "Age at Histological Diagnosis": [70, 65, 80, 55],
        "Gender": ["Female", "Male", "Male", "Female"],
        "Smoking status": ["Nonsmoker", "Current", "Former", "Former"],
    })
    out = labels_from_clinical(clinical)
    assert out["egfr"].tolist()[:2] == [1.0, 0.0] and out["egfr"].iloc[2:].isna().all()
    assert out["histology"].tolist() == ["adenocarcinoma", "squamous", "adenocarcinoma", "nos"]
    assert out["site"].tolist() == ["R01", "R01", "AMC", "R01"]
    assert out["smoking"].tolist() == ["never", "current", "former", "former"]


def test_modalities_from_digest():
    digest = pd.DataFrame({"Patient ID": ["R01-001"] * 3 + ["AMC-002"], "Modality": ["CT", "PT", "SEG", "CT"]})
    m = modalities_from_digest(digest).set_index("patient_id")
    assert m.loc["R01-001"].tolist() == [True, True, True]
    assert m.loc["AMC-002"].tolist() == [True, False, False]


def test_splits_cover_every_patient_once_and_spread_positives():
    cohort = _cohort()
    splits = make_splits(cohort, n_splits=5, seed=1)
    assert sorted(splits["patient_id"]) == sorted(cohort["patient_id"])
    assert set(splits["fold"]) == {0, 1, 2, 3, 4}
    per_fold = fold_summary(cohort, splits, "egfr")["positives"]
    assert per_fold.max() - per_fold.min() <= 2  # stratified
    assert make_splits(cohort, seed=1).equals(splits)  # reproducible


def test_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    perfect = binary_metrics(y, y.astype(float))
    assert perfect == {"auc": 1.0, "f1": 1.0, "sensitivity": 1.0, "specificity": 1.0}
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    lo, hi = bootstrap_ci(y, y + rng.normal(0, 1.0, 200), n_resamples=300)
    assert 0.5 < lo < hi < 1.0


def test_model_a_cross_validation_is_out_of_fold_and_learns_the_signal():
    cohort = _cohort(n=300)
    splits = make_splits(cohort, seed=0)
    oof = cross_validate(cohort, splits, "egfr")
    assert len(oof) == cohort["egfr"].notna().sum()  # only known labels, each once
    assert oof["patient_id"].is_unique
    assert binary_metrics(oof["y"], oof["prob"])["auc"] > 0.7


def test_exported_model_round_trip_and_app_predictor(tmp_path):
    model = fit_final(_cohort(n=300))
    model.cv_auc = {"egfr": [0.73, 0.63, 0.82], "kras": [0.55, 0.45, 0.65]}
    loaded = ClinicalModel.load(model.save(tmp_path / "clinical.json"))
    assert loaded.predict(70, "F", "never") == pytest.approx(model.predict(70, "F", "never"))
    assert loaded.predict(70, "F", "never")["egfr"] > loaded.predict(70, "M", "current")["egfr"]

    predictor = ClinicalPredictor(loaded)
    pred = predictor.predict(None, Localisation((0, 0, 0), "test"), ClinicalData(70, "feminin", "niciodată"))
    assert pred.placeholder_outputs == {"histology"} and not pred.is_placeholder
    assert pred.egfr == pytest.approx(loaded.predict(70, "F", "never")["egfr"])
    assert "nu e mai bun decât întâmplarea" in pred.sources["kras"]  # CI includes 0.5
    assert "întâmplarea" not in pred.sources["egfr"]
