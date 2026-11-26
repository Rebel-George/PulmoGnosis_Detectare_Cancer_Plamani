"""Patient-level, stratified cross-validation folds shared by every model.

The folds are generated once (fixed seed), saved as a small CSV under
configs/splits/ and reused, so all models are compared on exactly the same
patients. Stratification combines EGFR and KRAS status (including "unknown"),
so mutants of both kinds are spread evenly across folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_SPLITS = Path("configs/splits/nsclc_radiogenomics_5fold.csv")


def _status(value: float) -> str:
    return "?" if np.isnan(value) else ("M" if value == 1 else "W")


def stratification_key(cohort: pd.DataFrame, n_splits: int) -> pd.Series:
    """EGFR×KRAS status per patient; strata too small to spread over the folds are merged."""
    key = cohort["egfr"].map(_status) + cohort["kras"].map(_status)
    counts = key.value_counts()
    key = key.where(~key.isin(counts[counts < n_splits].index), "rare")
    if (key == "rare").sum() < n_splits:  # still too small: join the largest stratum
        key = key.replace("rare", key[key != "rare"].value_counts().idxmax())
    return key


def make_splits(cohort: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """One row per patient with its fold number (0 … n_splits-1)."""
    key = stratification_key(cohort, n_splits)
    folds = np.empty(len(cohort), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(cohort)), key)):
        folds[test_idx] = fold
    return pd.DataFrame({"patient_id": cohort["patient_id"].values, "fold": folds})


def save_splits(splits: pd.DataFrame, path: str | Path = DEFAULT_SPLITS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    return path


def load_splits(path: str | Path = DEFAULT_SPLITS) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str})


def fold_summary(cohort: pd.DataFrame, splits: pd.DataFrame, target: str) -> pd.DataFrame:
    """Patients with a known label and positives per fold, as the report must show."""
    df = cohort.merge(splits, on="patient_id")
    df = df[df[target].notna()]
    return df.groupby("fold")[target].agg(patients="count", positives="sum").astype(int)
