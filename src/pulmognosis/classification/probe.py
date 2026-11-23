"""Linear probe: frozen MedicalNet features + logistic regression (Model C, small-data variant).

Fine-tuning a 3D ResNet on ~140 patients is unstable, so the pretrained encoder is
used as a fixed feature extractor instead: every cube becomes one 512-value
vector, and a regularised logistic regression is cross-validated on those
vectors over the shared folds. The feature scaling is fitted on the training
fold only; the regularisation strength is fixed in advance (see
`cross_validate_probe`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pulmognosis.classification.dataset import load_cube

INPUT_NORMS = ("zscore", "unit")


def prepare_input(cube: np.ndarray, input_norm: str = "zscore", crop: int = 0) -> np.ndarray:
    """Cube (1, D, H, W) in [0, 1] → network input, optionally centre-cropped to `crop` voxels per side.

    "zscore" standardises each cube to mean 0 / std 1 (MedicalNet's own pre-training
    normalisation); "unit" keeps the [0, 1] scaling used for fine-tuning.
    """
    if crop:
        starts = [(s - crop) // 2 for s in cube.shape[1:]]
        cube = cube[:, starts[0]:starts[0] + crop, starts[1]:starts[1] + crop, starts[2]:starts[2] + crop]
    if input_norm == "zscore":
        cube = (cube - cube.mean()) / max(float(cube.std()), 1e-6)
    elif input_norm != "unit":
        raise ValueError(f"input_norm must be one of {INPUT_NORMS}")
    return np.ascontiguousarray(cube, dtype=np.float32)


@torch.no_grad()
def extract_features(paths: list[Path], encoder: torch.nn.Module, device: torch.device,
                     input_norm: str = "zscore", crop: int = 0) -> np.ndarray:
    """One feature vector per cube, from the encoder in eval mode (BatchNorm uses its stored statistics)."""
    encoder.eval()
    out = []
    for path in paths:
        x = torch.from_numpy(prepare_input(load_cube(path), input_norm, crop))[None].to(device)
        out.append(encoder(x).float().cpu().numpy()[0])
    return np.stack(out)


def cross_validate_probe(features: np.ndarray, y: np.ndarray, folds: np.ndarray, c: float = 1.0) -> pd.DataFrame:
    """Out-of-fold probabilities: index, fold, y, prob, threshold (0.5: the classes are weighted equally).

    The regularisation strength `c` is fixed in advance, not tuned per fold: folds tuned to very
    different strengths produce differently scaled probabilities, and pooling those out-of-fold
    probabilities into one AUC then scrambles the ranking (seen in testing: 0.56 vs. 0.68).
    """
    y = np.asarray(y).astype(int)
    rows = []
    for fold in sorted(np.unique(folds)):
        train, test = folds != fold, folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
        model.fit(features[train], y[train])
        rows.append(pd.DataFrame({
            "index": np.flatnonzero(test),
            "fold": fold,
            "y": y[test],
            "prob": model.predict_proba(features[test])[:, 1],
            "threshold": 0.5,
        }))
    return pd.concat(rows, ignore_index=True)
