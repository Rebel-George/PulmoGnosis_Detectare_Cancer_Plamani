"""Download the app's two held-out test patients (public NSCLC-Radiogenomics data, ~320 MB).

R01-080 (adenocarcinoma) and R01-007 (squamous) were drawn at random and left out of the
exported Models A and C, so the app's results on them are on unseen patients. For each, this
fetches the CT series the radiologist drew the tumour mask on, plus the mask (SEG), from the
NCI Imaging Data Commons into data/demo/<patient>/ (git-ignored). Load a patient folder in the
app's step 1 ("Folder local cu fișiere DICOM"); step 4 then starts from the mask's centre.

Usage (from the repository root; needs `pip install idc-index`):
    python scripts/download_demo_cases.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

PATIENTS = ("R01-080", "R01-007")
# The CT series each patient's SEG mask references (from the cube build log on the server).
CT_SERIES = (
    "1.3.6.1.4.1.14519.5.2.1.4334.1501.196193502776979291835885800180",  # R01-080
    "1.3.6.1.4.1.14519.5.2.1.4334.1501.273402029178462221323992286443",  # R01-007
)
DIR_TEMPLATE = "%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/demo")
    args = parser.parse_args()

    from idc_index import IDCClient

    client = IDCClient()
    index = client.index
    original_seg = (index["PatientID"].isin(PATIENTS) & (index["Modality"] == "SEG")
                    & index["analysis_result_id"].isna())  # the radiologists' masks, not the later AI ones
    selection = index[index["SeriesInstanceUID"].isin(CT_SERIES) | original_seg]
    print(selection[["PatientID", "Modality", "SeriesDescription", "series_size_MB"]].to_string(index=False))
    print(f"total {selection['series_size_MB'].sum():.0f} MB, into {Path(args.out).resolve()}")
    client.download_from_selection(seriesInstanceUID=list(selection["SeriesInstanceUID"]),
                                   downloadDir=args.out, dirTemplate=DIR_TEMPLATE)


if __name__ == "__main__":
    main()
