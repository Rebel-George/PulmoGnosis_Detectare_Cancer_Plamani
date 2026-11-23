"""Model A: cross-validate the clinical logistic regression, then export it for the app.

Usage (from the repository root, after scripts/build_cohort.py):
    python scripts/train_clinical.py

Writes:
    configs/splits/nsclc_radiogenomics_5fold.csv   shared folds (created once, then reused)
    runs/model_a_clinical/                         metrics.json, out-of-fold predictions, config
    models/clinical_lr.json                        model used by the app
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pulmognosis.classification.clinical import TARGETS, cross_validate, fit_final
from pulmognosis.evaluation.metrics import binary_metrics, bootstrap_ci
from pulmognosis.evaluation.splits import DEFAULT_SPLITS, fold_summary, load_splits, make_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="data/cohort/nsclc_radiogenomics.csv")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--out", default="runs/model_a_clinical")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--exclude", default="", help="comma-separated patient IDs left out of the exported model")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})
    if Path(args.splits).exists():
        splits = load_splits(args.splits)
    else:
        splits = make_splits(cohort, seed=args.seed)
        print(f"Created folds: {save_splits(splits, args.splits)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for target in TARGETS:
        oof = cross_validate(cohort, splits, target, c=args.C)
        oof.to_csv(out / f"oof_{target}.csv", index=False)
        pooled = binary_metrics(oof["y"], oof["prob"], oof["threshold"])
        lo, hi = bootstrap_ci(oof["y"].values, oof["prob"].values, n_resamples=args.bootstrap, seed=args.seed)
        per_fold = [binary_metrics(g["y"], g["prob"])["auc"] for _, g in oof.groupby("fold")]
        results[target] = {
            **pooled,
            "auc_ci95": [lo, hi],
            "auc_per_fold": per_fold,
            "auc_fold_mean": float(np.mean(per_fold)),
            "auc_fold_std": float(np.std(per_fold)),
            "patients": int(len(oof)),
            "positives": int(oof["y"].sum()),
            "per_fold": fold_summary(cohort, splits, target).reset_index().to_dict(orient="records"),
        }
        print(f"{target.upper()}: AUC {pooled['auc']:.3f} (95% CI {lo:.3f}–{hi:.3f}), "
              f"F1 {pooled['f1']:.3f}, sens {pooled['sensitivity']:.3f}, spec {pooled['specificity']:.3f}, "
              f"{len(oof)} patients / {int(oof['y'].sum())} positive")

    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Patients kept out of the final model (e.g. the app's held-out test cases); CV above uses everyone.
    held_out = [p for p in args.exclude.split(",") if p]
    model = fit_final(cohort[~cohort["patient_id"].isin(held_out)], c=args.C)
    if held_out:
        print(f"Final model fitted without {len(held_out)} held-out patients: {', '.join(held_out)}")
    model.cv_auc = {t: [r["auc"], *r["auc_ci95"]] for t, r in results.items()}
    model_path = model.save()
    print(f"Saved metrics to {out}/ and the app model to {model_path}")


if __name__ == "__main__":
    main()
