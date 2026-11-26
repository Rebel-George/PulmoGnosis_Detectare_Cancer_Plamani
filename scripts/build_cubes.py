"""Build the classifier's input cubes from downloaded NSCLC-Radiogenomics patients.

For every patient folder: find the SEG tumour mask → the CT series it was drawn
on → the mask's centroid in mm → a 128³ cube at 0.75 mm around it, saved as
data/processed/<dataset>/<patient>.npz (CT in HU, float16). PET is added in
Stage 3, after CT–CT registration (the PET is in a different study).

Usage (from the repository root):
    python scripts/build_cubes.py --raw ~/pulmognosis_data/raw/nsclc_radiogenomics \
                                  --out ~/pulmognosis_data/processed/nsclc_radiogenomics
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from pulmognosis.data.dicom import pick_series, scan_folder
from pulmognosis.data.seg import find_seg_files, read_seg
from pulmognosis.pipeline import MODEL_PATCH_SIZE, MODEL_SPACING_MM, extract_model_patch_mm, prepare_case


def build_one(patient_dir: Path, out_path: Path) -> dict:
    segs = find_seg_files(patient_dir)
    if not segs:
        raise ValueError("no SEG file")
    mask = read_seg(segs[0])

    series = scan_folder(patient_dir)
    ct = next((s for s in series if s.uid == mask.referenced_series_uid), None)
    how = "referenced by SEG"
    if ct is None:  # fall back to the largest CT in the SEG's study
        study_cts = [s for s in series if s.modality == "CT" and mask.study_uid and s.files[0].parent.parent.name == mask.study_uid]
        ct = pick_series(study_cts or series, "CT")
        how = "largest CT in SEG study" if study_cts else "largest CT"
    if ct is None:
        raise ValueError("no CT series")

    case = prepare_case(ct, None)
    cube, _ = extract_model_patch_mm(case, mask.center_mm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ct=cube.astype(np.float16),
        center_mm=mask.center_mm,
        extent_mm=mask.extent_mm,
        ct_spacing_mm=np.array(case.spacing_xyz),
        patch_spacing_mm=MODEL_SPACING_MM,
    )
    return {
        "ct_series": ct.uid,
        "ct_choice": how,
        "ct_slices": ct.num_slices,
        "ct_spacing_mm": "x".join(f"{v:.2f}" for v in case.spacing_xyz),
        "tumour_voxels": mask.n_voxels,
        "tumour_extent_mm": "x".join(f"{v:.0f}" for v in mask.extent_mm),
        "center_hu": float(cube[(MODEL_PATCH_SIZE // 2,) * 3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="folder with one sub-folder per patient")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw, out = Path(args.raw).expanduser(), Path(args.out).expanduser()
    patients = sorted(p for p in raw.iterdir() if p.is_dir())
    rows = []
    for i, patient_dir in enumerate(patients, 1):
        out_path = out / f"{patient_dir.name}.npz"
        row = {"patient_id": patient_dir.name, "status": "ok", "error": ""}
        if out_path.exists() and not args.overwrite:
            row["status"] = "exists"
        else:
            try:
                row.update(build_one(patient_dir, out_path))
            except Exception as exc:  # keep going; failures are listed in the log
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
        rows.append(row)
        print(f"[{i}/{len(patients)}] {patient_dir.name}: {row['status']} {row['error']}")

    log = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    log.to_csv(out / "build_log.csv", index=False)
    print(log["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
