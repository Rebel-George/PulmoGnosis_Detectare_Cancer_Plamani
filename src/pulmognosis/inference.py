"""Model interface used by the app, plus a placeholder implementation.

The app only talks to `Predictor`. Stage 1 (U-Net) and Stage 2 (late-fusion
classifier) will provide a real implementation with the same two methods,
so swapping the placeholder out does not change the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import ndimage

from pulmognosis.pipeline import Case, ClinicalData
from pulmognosis.preprocessing.spatial import to_array

HISTOLOGY_CLASSES = ("adenocarcinom", "carcinom scuamos", "carcinom cu celule mici")

# Label prevalence counted from the TCIA metadata (see the report's data tables).
# Histology: Lung-PET-CT-Dx + NSCLC-Radiogenomics; mutations: NSCLC-Radiogenomics.
_HISTOLOGY_COUNTS = {"adenocarcinom": 423, "carcinom scuamos": 96, "carcinom cu celule mici": 38}
_EGFR_RATE = 43 / 172
_KRAS_RATE = 38 / 171


@dataclass
class Localisation:
    center_zyx: tuple[int, int, int]
    method: str


@dataclass
class Prediction:
    histology: dict[str, float]
    egfr: float
    kras: float
    is_placeholder: bool
    note: str


class Predictor(Protocol):
    def localise(self, case: Case) -> Localisation: ...

    def predict(self, case: Case, location: Localisation, clinical: ClinicalData) -> Prediction: ...


class PlaceholderPredictor:
    """Stand-in until the trained models exist. Its outputs are NOT predictions.

    - Localisation: the hottest PET region (smoothed SUV maximum) when PET is
      available, otherwise the volume centre for the user to adjust by hand.
    - Classification: the label prevalence in the training datasets, the same
      for every patient.
    """

    def localise(self, case: Case) -> Localisation:
        if case.has_pet:
            suv = ndimage.gaussian_filter(to_array(case.pet), sigma=2)
            center = np.unravel_index(int(np.argmax(suv)), suv.shape)
            return Localisation(tuple(int(c) for c in center), "punctul cu SUV maxim (euristică)")
        shape = to_array(case.ct).shape
        return Localisation(tuple(s // 2 for s in shape), "centrul volumului (ajustați manual)")

    def predict(self, case: Case, location: Localisation, clinical: ClinicalData) -> Prediction:
        total = sum(_HISTOLOGY_COUNTS.values())
        return Prediction(
            histology={k: v / total for k, v in _HISTOLOGY_COUNTS.items()},
            egfr=_EGFR_RATE,
            kras=_KRAS_RATE,
            is_placeholder=True,
            note="Valori de referință: prevalența din seturile de antrenare, identică pentru toți pacienții. "
            "Nu sunt predicții — modelul nu este încă antrenat.",
        )
