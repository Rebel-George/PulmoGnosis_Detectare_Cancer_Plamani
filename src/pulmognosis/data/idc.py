"""Selective downloads from NCI Imaging Data Commons (the same data as TCIA) with `idc-index`.

Only the series the project uses are selected, so the download fits the
storage budget (see PLAN.md §3.6). The index is a pandas DataFrame with one
row per series (collection_id, PatientID, StudyInstanceUID, SeriesInstanceUID,
Modality, SeriesDescription, series_size_MB, ...).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NSCLC_RADIOGENOMICS = "nsclc_radiogenomics"
# Patient / study / series folders: lets the code tell the diagnostic CT study from the PET/CT study.
DIR_TEMPLATE = "%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"
_UNCORRECTED_PET = r"NO_AC|NOAC|NON_AC|\bNAC\b"


def is_original_seg(rows: pd.DataFrame) -> pd.Series:
    """True for the collection's own expert tumour masks.

    IDC also hosts masks added later by analysis projects (e.g. the AI lung + tumour
    segmentations from `bamf_aimi_annotations`); those have an `analysis_result_id`
    and outline the lungs as well, so a cube centred on them would miss the tumour.
    """
    is_seg = rows["Modality"] == "SEG"
    if "analysis_result_id" in rows:
        return is_seg & rows["analysis_result_id"].isna()
    return is_seg & ~rows["SeriesDescription"].fillna("").str.upper().str.startswith("AIMI")


def select_nsclc_radiogenomics(index: pd.DataFrame, patients: list[str] | None = None) -> pd.DataFrame:
    """Series needed for the R01 patients that have an original expert tumour mask (SEG).

    Keeps every CT (the diagnostic CT the mask is drawn on and the low-dose CT of
    the PET/CT study, needed for registration), the original SEG, and the
    attenuation-corrected PET; drops AI-generated masks and uncorrected PET
    reconstructions, which cannot be converted to SUV. `patients` restricts the
    selection further (e.g. a small-data subset).
    """
    r = index[(index["collection_id"] == NSCLC_RADIOGENOMICS) & index["PatientID"].str.startswith("R01")]
    original_seg = is_original_seg(r)
    with_seg = set(r.loc[original_seg, "PatientID"])
    if patients is not None:
        with_seg &= set(patients)
    r, original_seg = r[r["PatientID"].isin(with_seg)], original_seg[r["PatientID"].isin(with_seg)]
    desc = r["SeriesDescription"].fillna("").str.upper()
    keep = (r["Modality"] == "CT") | original_seg | ((r["Modality"] == "PT") & ~desc.str.contains(_UNCORRECTED_PET))
    return r[keep].copy()


def size_report(selection: pd.DataFrame) -> pd.DataFrame:
    """GB and series count per modality, plus the total."""
    by_mod = selection.groupby("Modality").agg(series=("SeriesInstanceUID", "count"), GB=("series_size_MB", "sum"))
    by_mod["GB"] = by_mod["GB"] / 1024
    by_mod.loc["TOTAL"] = [by_mod["series"].sum(), by_mod["GB"].sum()]
    return by_mod.round({"GB": 1})


def download(client, selection: pd.DataFrame, out_dir: str | Path) -> None:
    """Download the selected series (re-running skips what is already there)."""
    client.download_from_selection(
        seriesInstanceUID=list(selection["SeriesInstanceUID"]),
        downloadDir=str(out_dir),
        dirTemplate=DIR_TEMPLATE,
    )
