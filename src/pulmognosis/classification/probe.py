"""Linear probe: frozen MedicalNet features + logistic regression (Model C, small-data variant).

Fine-tuning a 3D ResNet on ~140 patients is unstable, so the pretrained encoder is
used as a fixed feature extractor instead: every cube becomes one 512-value
vector, and a regularised logistic regression is cross-validated on those
vectors over the shared folds. The feature scaling is fitted on the training
fold only; the regularisation strength is fixed in advance (see
`cross_validate_probe`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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

    "zscore" standardises each cube to mean 0 / std 1; "unit" keeps the [0, 1]
    scaling used for fine-tuning (the ablation found no difference between the two).
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
def cube_features(cube: np.ndarray, encoder: torch.nn.Module, device: torch.device,
                  input_norm: str = "zscore", crop: int = 0) -> np.ndarray:
    """The feature vector of one cube (1, D, H, W) in [0, 1], from the encoder in eval mode
    (BatchNorm uses its stored statistics). Training and the app both go through here."""
    encoder.eval()
    x = torch.from_numpy(prepare_input(cube, input_norm, crop))[None].to(device)
    return encoder(x).float().cpu().numpy()[0]


def extract_features(paths: list[Path], encoder: torch.nn.Module, device: torch.device,
                     input_norm: str = "zscore", crop: int = 0) -> np.ndarray:
    """One feature vector per saved cube."""
    return np.stack([cube_features(load_cube(path), encoder, device, input_norm, crop) for path in paths])


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
        model = _pipeline(c).fit(features[train], y[train])
        rows.append(pd.DataFrame({
            "index": np.flatnonzero(test),
            "fold": fold,
            "y": y[test],
            "prob": model.predict_proba(features[test])[:, 1],
            "threshold": 0.5,
        }))
    return pd.concat(rows, ignore_index=True)


def _pipeline(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


@dataclass
class ProbeModel:
    """Model C as exported for the app: how to make the features, plus the fitted scaling and regression.

    Stored as a small JSON file (like Model A), so the app needs no scikit-learn model object.
    The classes are weighted equally in training, so the output is a score centred on 0.5,
    not a calibrated probability.
    """

    target: str
    classes: list[str]  # [negative, positive] as shown in the app
    pretrained: str
    input_norm: str
    crop: int
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    c: float
    n_patients: int
    n_positive: int
    cv_auc: list[float] = field(default_factory=list)  # [auc, ci_low, ci_high]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Score of the positive class for each feature vector (n, d) or one vector (d,)."""
        f = np.atleast_2d(np.asarray(features, dtype=float))
        logit = ((f - np.array(self.mean)) / np.array(self.scale)) @ np.array(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-logit))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbeModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_final(features: np.ndarray, y: np.ndarray, *, c: float = 1.0, **settings) -> ProbeModel:
    """Fit the probe on every patient (after cross-validation has estimated its performance)."""
    y = np.asarray(y).astype(int)
    pipe = _pipeline(c).fit(features, y)
    scaler, lr = pipe[0], pipe[-1]
    return ProbeModel(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=lr.coef_[0].tolist(),
                      intercept=float(lr.intercept_[0]), c=c, n_patients=int(len(y)), n_positive=int(y.sum()),
                      **settings)
