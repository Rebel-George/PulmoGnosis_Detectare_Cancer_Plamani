"""Model interface used by the app, plus a placeholder implementation.

The app only talks to `Predictor`. Stage 1 (U-Net) and Stage 2 (late-fusion
classifier) will provide a real implementation with the same two methods,
so swapping the placeholder out does not change the app. Each `Prediction`
says where every output comes from and which outputs are still placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import ndimage

from pulmognosis.classification.clinical import ClinicalModel
from pulmognosis.pipeline import SEX_CHOICES, SMOKING_CHOICES, Case, ClinicalData
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
    center_mm: tuple[float, float, float] | None = None  # exact centre when known (a SEG centroid), else from the voxel


def seg_localisation(folder: str | Path, ct_uid: str, case: Case) -> Localisation | None:
    """The tumour centre from a radiologist's SEG mask in the loaded folder, if one belongs to this CT.

    This is how the training cubes were centred, so a case loaded with its SEG gets the same input
    the model was trained on. A SEG drawn on another CT series is ignored (different coordinates).
    """
    from pulmognosis.data.seg import find_seg_files, read_seg

    for path in find_seg_files(folder):
        try:
            mask = read_seg(path)
        except ValueError:
            continue
        if mask.referenced_series_uid in (None, ct_uid):
            center_mm = tuple(float(v) for v in mask.center_mm)
            return Localisation(case.mm_to_index(center_mm), "centrul măștii tumorale (SEG) desenate de radiolog",
                                center_mm)
    return None


OUTPUTS = ("histology", "egfr", "kras")


@dataclass
class Prediction:
    histology: dict[str, float]
    egfr: float
    kras: float
    note: str
    placeholder_outputs: frozenset[str] = frozenset(OUTPUTS)
    sources: dict[str, str] = field(default_factory=dict)  # output → where the value comes from

    @property
    def is_placeholder(self) -> bool:
        """True when no output comes from a trained model."""
        return self.placeholder_outputs == frozenset(OUTPUTS)


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
        if case.pet_native is not None:
            # Search on the coarse native PET grid (fast), then map the point onto the CT.
            pet = case.pet_native
            suv = ndimage.gaussian_filter(to_array(pet), sigma=1)
            z, y, x = np.unravel_index(int(np.argmax(suv)), suv.shape)
            point_mm = pet.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            return Localisation(case.mm_to_index(point_mm), "punctul cu SUV maxim (euristică)")
        shape = to_array(case.ct).shape
        return Localisation(tuple(s // 2 for s in shape), "centrul volumului (ajustați manual)")

    def predict(self, case: Case, location: Localisation, clinical: ClinicalData) -> Prediction:
        total = sum(_HISTOLOGY_COUNTS.values())
        return Prediction(
            histology={k: v / total for k, v in _HISTOLOGY_COUNTS.items()},
            egfr=_EGFR_RATE,
            kras=_KRAS_RATE,
            note="Valori de referință: prevalența din seturile de antrenare, identică pentru toți pacienții. "
            "Nu sunt predicții — modelul nu este încă antrenat.",
            sources={k: "prevalența din seturile de date (placeholder)" for k in OUTPUTS},
        )


# App labels → codes used by the cohort table and the clinical model.
_SEX_CODE = dict(zip(SEX_CHOICES, ("M", "F")))
_SMOKING_CODE = dict(zip(SMOKING_CHOICES, ("never", "former", "current")))


class ClinicalPredictor(PlaceholderPredictor):
    """Model A in the app: EGFR / KRAS from age, sex and smoking (logistic regression).

    Histology and localisation are still the placeholder ones until the image
    models are trained.
    """

    def __init__(self, model: ClinicalModel):
        self.model = model

    @classmethod
    def from_file(cls, path: str | Path) -> "ClinicalPredictor":
        return cls(ClinicalModel.load(path))

    def _source(self, target: str) -> str:
        n, pos = self.model.n_patients[target], self.model.n_positive[target]
        text = f"model clinic (regresie logistică pe vârstă, gen, fumat), antrenat pe {n} pacienți ({pos} mutanți)"
        if target in self.model.cv_auc:
            auc, lo, hi = self.model.cv_auc[target]
            text += f"; AUC în validare încrucișată {auc:.2f} (IÎ 95% {lo:.2f}–{hi:.2f})"
            if lo <= 0.5:
                text += " — nu e mai bun decât întâmplarea"
        return text

    def predict(self, case: Case, location: Localisation, clinical: ClinicalData) -> Prediction:
        base = super().predict(case, location, clinical)
        probs = self.model.predict(clinical.age, _SEX_CODE[clinical.sex], _SMOKING_CODE[clinical.smoking])
        return Prediction(
            histology=base.histology,
            egfr=probs["egfr"],
            kras=probs["kras"],
            note="EGFR și KRAS vin din modelul clinic (doar vârstă, gen și fumat, fără imagini). "
            "Tipul histologic este încă valoare de referință, identică pentru toți pacienții.",
            placeholder_outputs=frozenset({"histology"}),
            sources={"histology": base.sources["histology"], "egfr": self._source("egfr"), "kras": self._source("kras")},
        )


class HistologyProbe:
    """Model C in the app: the frozen 3D ResNet-18 turns the CT cube into 512 features, and the
    exported logistic regression scores adenocarcinoma vs. squamous from them.

    torch and MONAI are imported only when the first cube is scored; with `pretrained="medicalnet"`
    the first use downloads the MedicalNet weights through MONAI (then cached).
    """

    def __init__(self, model):
        self.model = model  # pulmognosis.classification.probe.ProbeModel
        self._encoder = None

    @classmethod
    def from_file(cls, path: str | Path) -> "HistologyProbe":
        from pulmognosis.classification.probe import ProbeModel

        return cls(ProbeModel.load(path))

    def _get_encoder(self):
        if self._encoder is None:
            import torch

            from pulmognosis.classification.model3d import ResNet3DClassifier

            torch.manual_seed(0)  # only matters for pretrained="none" (tests)
            self._encoder = ResNet3DClassifier(in_channels=1, pretrained=self.model.pretrained).encoder.eval()
        return self._encoder

    def score(self, ct_cube_hu: np.ndarray) -> float:
        """Score of the positive class (squamous) for one CT cube in HU, (z, y, x)."""
        import torch

        from pulmognosis.classification.probe import cube_features
        from pulmognosis.preprocessing.intensity import normalize_ct

        cube = normalize_ct(np.asarray(ct_cube_hu, dtype=np.float32))[None]
        feats = cube_features(cube, self._get_encoder(), torch.device("cpu"), self.model.input_norm, self.model.crop)
        return float(self.model.predict_proba(feats)[0])

    def source(self) -> str:
        m = self.model
        text = (f"model de imagine C (3D ResNet-18 MedicalNet fix + regresie logistică, doar CT), "
                f"antrenat pe {m.n_patients} pacienți ({m.n_positive} scuamoși)")
        if m.cv_auc:
            auc, lo, hi = m.cv_auc
            text += f"; AUC în validare încrucișată {auc:.2f} (IÎ 95% {lo:.2f}–{hi:.2f})"
        return (text + ". Doar adenocarcinom vs. scuamos (fără celule mici); clasele au fost ponderate egal, "
                "deci valoarea este un scor, nu o probabilitate calibrată. La antrenare cubul a fost centrat pe "
                "masca radiologului, aici pe centrul ales în pasul 4.")


class ImagePredictor(ClinicalPredictor):
    """Model A for EGFR / KRAS plus Model C for histology; localisation is still the heuristic one."""

    def __init__(self, model: ClinicalModel, histology: HistologyProbe):
        super().__init__(model)
        self.histology = histology

    def predict(self, case: Case, location: Localisation, clinical: ClinicalData) -> Prediction:
        from pulmognosis.pipeline import extract_model_patch, extract_model_patch_mm

        base = super().predict(case, location, clinical)
        if location.center_mm is not None:  # exact SEG centre, as in training
            ct_cube, _ = extract_model_patch_mm(case, location.center_mm)
        else:
            ct_cube, _ = extract_model_patch(case, location.center_zyx)
        p = self.histology.score(ct_cube)
        negative, positive = self.histology.model.classes
        return Prediction(
            histology={negative: 1.0 - p, positive: p},
            egfr=base.egfr,
            kras=base.kras,
            note="Tipul histologic vine din modelul de imagine C (doar CT, cubul de 96 mm din pasul 4); "
            "EGFR și KRAS vin din modelul clinic A (vârstă, gen, fumat). Ambele sunt modele slabe: vezi AUC-ul de sub fiecare.",
            placeholder_outputs=frozenset(),
            sources={**base.sources, "histology": self.histology.source()},
        )
