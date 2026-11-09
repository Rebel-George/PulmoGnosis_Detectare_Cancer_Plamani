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


def select_nsclc_radiogenomics(index: pd.DataFrame, patients: list[str] | None = None) -> pd.DataFrame:
    """Series needed for the R01 patients that have a tumour mask (SEG).

    Keeps every CT (the diagnostic CT the mask is drawn on and the low-dose CT of
    the PET/CT study, needed for registration), the SEG, and the attenuation-corrected
    PET; drops uncorrected PET reconstructions, which cannot be converted to SUV.
    `patients` restricts the selection further (e.g. a small-data subset).
    """
    r = index[(index["collection_id"] == NSCLC_RADIOGENOMICS) & index["PatientID"].str.startswith("R01")]
    with_seg = set(r.loc[r["Modality"] == "SEG", "PatientID"])
    if patients is not None:
        with_seg &= set(patients)
    r = r[r["PatientID"].isin(with_seg)]
    desc = r["SeriesDescription"].fillna("").str.upper()
    keep = r["Modality"].isin(["CT", "SEG"]) | ((r["Modality"] == "PT") & ~desc.str.contains(_UNCORRECTED_PET))
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
