"""Model C, linear-probe variant: frozen MedicalNet features + logistic regression.

The pretrained 3D ResNet-18 is not trained at all: each cube is turned into a
512-value feature vector once (cached in <out>/features.npz), then a
class-balanced logistic regression (fixed regularisation, as in Model A) is
cross-validated on the shared folds. Same metrics and output
files as train_model3d.py, so the results table picks it up the same way.

Usage (from the repository root; a GPU makes feature extraction take seconds):
    python scripts/probe_model3d.py --cubes ~/pulmognosis_data/processed/nsclc_radiogenomics \
        --out runs/model_c_probe
    # tumour-focused variant: centre 64³ voxels (48 mm) instead of the whole 96 mm cube
    python scripts/probe_model3d.py ... --crop 64 --out runs/model_c_probe_crop64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pulmognosis.classification.dataset import labelled_patients
from pulmognosis.classification.model3d import ResNet3DClassifier
from pulmognosis.classification.probe import INPUT_NORMS, cross_validate_probe, extract_features
from pulmognosis.evaluation.metrics import binary_metrics, bootstrap_ci
from pulmognosis.evaluation.splits import DEFAULT_SPLITS, load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cubes", required=True)
    parser.add_argument("--cohort", default="data/cohort/nsclc_radiogenomics.csv")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--target", default="histology", choices=["histology", "egfr", "kras"])
    parser.add_argument("--pretrained", default="medicalnet", choices=["medicalnet", "none"])
    parser.add_argument("--input-norm", default="zscore", choices=INPUT_NORMS)
    parser.add_argument("--crop", type=int, default=0, help="centre crop in voxels per side (0 = whole cube)")
    parser.add_argument("--c", type=float, default=1.0,
                        help="inverse regularisation strength, fixed in advance (same default as Model A)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})
    df = labelled_patients(cohort, Path(args.cubes).expanduser(), args.target)
    df = df.merge(load_splits(args.splits), on="patient_id").reset_index(drop=True)
    print(f"{len(df)} patients, {int(df['y'].sum())} positive; device {device}; pretrained {args.pretrained}; "
          f"input {args.input_norm}; crop {args.crop or 'none'}")

    cache = out / "features.npz"
    settings = {"pretrained": args.pretrained, "input_norm": args.input_norm, "crop": args.crop}
    features = None
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if list(z["patient_id"]) == list(df["patient_id"]) and json.loads(str(z["settings"])) == settings:
                features = z["features"]
                print(f"features loaded from {cache}")
    if features is None:
        encoder = ResNet3DClassifier(in_channels=1, pretrained=args.pretrained).encoder.to(device)
        features = extract_features(list(df["path"]), encoder, device, args.input_norm, args.crop)
        np.savez(cache, features=features, patient_id=df["patient_id"].to_numpy(dtype=str), settings=json.dumps(settings))
        print(f"features {features.shape} saved to {cache}")

    oof = cross_validate_probe(features, df["y"].to_numpy(), df["fold"].to_numpy(), c=args.c)
    oof.insert(0, "patient_id", df["patient_id"].to_numpy()[oof["index"]])
    oof = oof.drop(columns="index")
    oof.to_csv(out / "oof.csv", index=False)

    pooled = binary_metrics(oof["y"], oof["prob"], oof["threshold"])
    lo, hi = bootstrap_ci(oof["y"].to_numpy(), oof["prob"].to_numpy(), n_resamples=args.bootstrap, seed=args.seed)
    per_fold = {int(f): binary_metrics(g["y"], g["prob"])["auc"] for f, g in oof.groupby("fold") if g["y"].nunique() == 2}
    metrics = {
        **pooled, "auc_ci95": [lo, hi], "auc_per_fold": per_fold,
        "auc_mean_of_folds": float(np.mean(list(per_fold.values()))),
        "patients": len(oof), "positives": int(oof["y"].sum()),
        "per_fold_counts": oof.groupby("fold")["y"].agg(patients="count", positives="sum").reset_index().to_dict("records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({**vars(args), "device": str(device)}, indent=2), encoding="utf-8")
    print(f"{args.target}: AUC {pooled['auc']:.3f} (95% CI {lo:.3f}–{hi:.3f}), F1 {pooled['f1']:.3f}, "
          f"sens {pooled['sensitivity']:.3f}, spec {pooled['specificity']:.3f}, "
          f"{len(oof)} patients / {int(oof['y'].sum())} positive")
    print("per fold:", {k: round(v, 3) for k, v in per_fold.items()},
          f"| mean of folds {metrics['auc_mean_of_folds']:.3f}")


if __name__ == "__main__":
    main()
