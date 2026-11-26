"""Cohort table for NSCLC-Radiogenomics: one row per patient with modalities and labels.

Built from the two TCIA metadata files (clinical CSV and NBIA series digest).
Labels use 1 / 0 for mutant / wild type and NaN when unknown or not collected,
so missing labels are never mistaken for negatives.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CLINICAL_CSV = "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
SERIES_DIGEST = "NSCLC_Radiogenomics-6-1-21-Version-4-nbia-digest.xlsx"

_MUTATION = {"Mutant": 1.0, "Wildtype": 0.0}  # anything else (Unknown, Not collected) → NaN
_HISTOLOGY = {
    "Adenocarcinoma": "adenocarcinoma",
    "Squamous cell carcinoma": "squamous",
    "NSCLC NOS (not otherwise specified)": "nos",
}
_SMOKING = {"Nonsmoker": "never", "Former": "former", "Current": "current"}
_SEX = {"Male": "M", "Female": "F"}

COLUMNS = [
    "patient_id", "site", "has_ct", "has_pet", "has_seg",
    "histology", "egfr", "kras", "age", "sex", "smoking",
]


def labels_from_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    """Normalise the clinical CSV into tidy label and covariate columns."""
    out = pd.DataFrame({
        "patient_id": clinical["Case ID"].astype(str).str.strip(),
        "histology": clinical["Histology "].map(_HISTOLOGY),
        "egfr": clinical["EGFR mutation status"].map(_MUTATION).astype(float),
        "kras": clinical["KRAS mutation status"].map(_MUTATION).astype(float),
        "age": pd.to_numeric(clinical["Age at Histological Diagnosis"], errors="coerce"),
        "sex": clinical["Gender"].map(_SEX),
        "smoking": clinical["Smoking status"].map(_SMOKING),
    })
    out["site"] = out["patient_id"].str.split("-").str[0]
    return out


def modalities_from_digest(digest: pd.DataFrame) -> pd.DataFrame:
    """Which modalities (CT, PET, SEG) each patient has in the image archive."""
    mods = digest.groupby("Patient ID")["Modality"].apply(set)
    return pd.DataFrame({
        "patient_id": mods.index.astype(str),
        "has_ct": [("CT" in m) for m in mods],
        "has_pet": [("PT" in m) for m in mods],
        "has_seg": [("SEG" in m) for m in mods],
    })


def build_cohort(metadata_dir: str | Path) -> pd.DataFrame:
    metadata_dir = Path(metadata_dir)
    clinical = pd.read_csv(metadata_dir / CLINICAL_CSV)
    digest = pd.read_excel(metadata_dir / SERIES_DIGEST)

    cohort = labels_from_clinical(clinical).merge(modalities_from_digest(digest), on="patient_id", how="left")
    for col in ("has_ct", "has_pet", "has_seg"):
        cohort[col] = cohort[col].fillna(False).astype(bool)
    return cohort[COLUMNS].sort_values("patient_id").reset_index(drop=True)


def summarise(cohort: pd.DataFrame) -> dict[str, object]:
    """Counts used to sanity-check the cohort against the numbers in PLAN.md."""
    seg = cohort[cohort["has_seg"]]
    known = lambda s: int(s.notna().sum())
    pos = lambda s: int((s == 1).sum())
    return {
        "patients": len(cohort),
        "with_pet": int(cohort["has_pet"].sum()),
        "with_seg": len(seg),
        "egfr_known": known(cohort["egfr"]), "egfr_mutant": pos(cohort["egfr"]),
        "kras_known": known(cohort["kras"]), "kras_mutant": pos(cohort["kras"]),
        "histology": cohort["histology"].value_counts().to_dict(),
        "seg_histology": seg["histology"].value_counts().to_dict(),
    }
