"""Model A: logistic regression on age, sex and smoking status → EGFR / KRAS mutation.

The clinical baseline every image model must beat. Age is standardised with
statistics from the training patients only. The regression is unweighted, so
its outputs are real probabilities (shown as such in the app); because mutants
are a minority, the yes/no decision threshold is the mutant rate of the
training patients rather than 0.5. The fitted model is exported as a small JSON
file so the app can use it without scikit-learn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ("age_z", "sex_female", "smoking_former", "smoking_current")  # never-smoking male is the baseline
TARGETS = ("egfr", "kras")
DEFAULT_MODEL_PATH = Path("models/clinical_lr.json")


def design_matrix(age, sex, smoking, age_mean: float, age_std: float) -> np.ndarray:
    """Feature matrix in FEATURES order from cohort codes (sex M/F, smoking never/former/current)."""
    age, sex, smoking = (np.atleast_1d(np.asarray(v)) for v in (age, sex, smoking))
    return np.column_stack([
        (age.astype(float) - age_mean) / age_std,
        (sex == "F").astype(float),
        (smoking == "former").astype(float),
        (smoking == "current").astype(float),
    ])


def _fit(train: pd.DataFrame, target: str, c: float) -> tuple[LogisticRegression, float, float]:
    age_mean, age_std = float(train["age"].mean()), float(train["age"].std())
    x = design_matrix(train["age"], train["sex"], train["smoking"], age_mean, age_std)
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(x, train[target].astype(int))
    return model, age_mean, age_std


def cross_validate(cohort: pd.DataFrame, splits: pd.DataFrame, target: str, c: float = 1.0) -> pd.DataFrame:
    """Out-of-fold probabilities for every patient with a known label: patient_id, fold, y, prob, threshold.

    `threshold` is the mutant rate of that fold's training patients (the decision cut-off).
    """
    df = cohort.merge(splits, on="patient_id")
    df = df[df[target].notna()]
    rows = []
    for fold in sorted(df["fold"].unique()):
        train, test = df[df["fold"] != fold], df[df["fold"] == fold]
        model, age_mean, age_std = _fit(train, target, c)
        x = design_matrix(test["age"], test["sex"], test["smoking"], age_mean, age_std)
        rows.append(pd.DataFrame({
            "patient_id": test["patient_id"].values,
            "fold": fold,
            "y": test[target].astype(int).values,
            "prob": model.predict_proba(x)[:, 1],
            "threshold": float(train[target].mean()),
        }))
    return pd.concat(rows, ignore_index=True)


@dataclass
class ClinicalModel:
    """The exported model: one logistic regression per target, plus the age scaling."""

    age_mean: float
    age_std: float
    weights: dict[str, dict[str, float]]  # target → {"intercept": b, feature: w, ...}
    n_patients: dict[str, int]
    n_positive: dict[str, int]
    cv_auc: dict[str, list[float]] = field(default_factory=dict)  # target → [auc, ci_low, ci_high]

    def predict(self, age: float, sex: str, smoking: str) -> dict[str, float]:
        x = design_matrix(age, sex, smoking, self.age_mean, self.age_std)[0]
        out = {}
        for target, w in self.weights.items():
            logit = w["intercept"] + sum(w[f] * v for f, v in zip(FEATURES, x))
            out[target] = float(1 / (1 + np.exp(-logit)))
        return out

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "ClinicalModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_final(cohort: pd.DataFrame, c: float = 1.0) -> ClinicalModel:
    """Fit on every patient with a known label (after cross-validation has estimated performance)."""
    age_mean, age_std = float(cohort["age"].mean()), float(cohort["age"].std())
    weights, n_patients, n_positive = {}, {}, {}
    for target in TARGETS:
        known = cohort[cohort[target].notna()]
        x = design_matrix(known["age"], known["sex"], known["smoking"], age_mean, age_std)
        model = LogisticRegression(C=c, max_iter=1000).fit(x, known[target].astype(int))
        weights[target] = {"intercept": float(model.intercept_[0]), **dict(zip(FEATURES, map(float, model.coef_[0])))}
        n_patients[target] = len(known)
        n_positive[target] = int(known[target].sum())
    return ClinicalModel(age_mean, age_std, weights, n_patients, n_positive)
