"""Model C: 3D ResNet-18 on the CT cubes, cross-validated on the shared folds.

Target for Stage 2: histology, adenocarcinoma (0) vs squamous (1); NOS excluded.
For each fold: train on the other folds (15% of them held out for early stopping),
predict the test fold. Then a final model is trained on all patients for the
median number of epochs found in the folds, and saved for the app.

Usage (from the repository root, on a GPU machine):
    python scripts/train_model3d.py --cubes ~/pulmognosis_data/processed/nsclc_radiogenomics \
        --out runs/model_c_histology_all --pretrained medicalnet
    # small-data variant (Stage 2): a stratified subset of 40 patients
    python scripts/train_model3d.py ... --subset 40 --out runs/model_c_histology_40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pulmognosis.classification.dataset import CubeDataset
from pulmognosis.classification.model3d import ResNet3DClassifier
from pulmognosis.evaluation.metrics import binary_metrics, bootstrap_ci
from pulmognosis.evaluation.splits import DEFAULT_SPLITS, load_splits

HISTOLOGY_LABEL = {"adenocarcinoma": 0.0, "squamous": 1.0}


def labelled_patients(cohort: pd.DataFrame, cubes: Path, target: str) -> pd.DataFrame:
    df = cohort.copy()
    if target == "histology":
        df["y"] = df["histology"].map(HISTOLOGY_LABEL)
    else:  # egfr / kras (later stages)
        df["y"] = df[target]
    df["path"] = [cubes / f"{p}.npz" for p in df["patient_id"]]
    return df[df["y"].notna() & df["path"].map(Path.exists)].reset_index(drop=True)


def make_loader(df: pd.DataFrame, args, augment: bool, shuffle: bool, seed: int) -> DataLoader:
    ds = CubeDataset(list(df["path"]), list(df["y"]), augment=augment, seed=seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x.to(device)).float()).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, args, device, seed: int, epochs: int,
          augment: bool = True, on_epoch=None):
    """Train one model. With `val_df`, keeps the weights of the epoch with the lowest validation loss.

    `on_epoch(model, row)` is called after every epoch and may add entries to the logged row.
    """
    torch.manual_seed(seed)
    model = ResNet3DClassifier(in_channels=1, pretrained=args.pretrained).to(device)
    pos = float(train_df["y"].sum())
    pos_weight = torch.tensor([(len(train_df) - pos) / max(pos, 1.0)], device=device)  # balance classes in the loss
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = make_loader(train_df, args, augment=augment, shuffle=True, seed=seed)
    val_loader = make_loader(val_df, args, augment=False, shuffle=False, seed=seed) if val_df is not None else None

    history, best = [], (np.inf, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
        row = {"epoch": epoch, "train_loss": total / len(train_df)}
        if on_epoch is not None:
            on_epoch(model, row)
        if val_loader is not None:
            p = predict(model, val_loader, device)
            y_val = val_df["y"].to_numpy()
            val_loss = float(loss_fn(torch.logit(torch.tensor(p).clamp(1e-6, 1 - 1e-6)).to(device),
                                     torch.tensor(y_val, device=device)).item())
            row["val_loss"] = val_loss
            if val_loss < best[0]:
                best = (val_loss, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            elif epoch - best[1] >= args.patience:
                history.append(row)
                break
        history.append(row)
        print(f"  epoch {epoch:3d}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    return model, history, best[1] or epochs


def overfit_check(df: pd.DataFrame, args, device, out: Path) -> None:
    """Fit a few patients with nothing in the way: if the training loss does not fall, the bug is in
    the data or model path, not in the amount of data."""
    if args.overfit < len(df):
        df, _ = train_test_split(df, train_size=args.overfit, stratify=df["y"], random_state=args.seed)
    print(f"overfit check: {len(df)} patients, {int(df['y'].sum())} positive; lr {args.lr}; device {device}")
    eval_loader = make_loader(df, args, augment=False, shuffle=False, seed=args.seed)

    def report(model, row):
        p = predict(model, eval_loader, device)
        row["train_auc_eval_mode"] = binary_metrics(df["y"], p)["auc"]
        row["prob_spread"] = float(p.max() - p.min())
        model.train()

    _, history, _ = train(df, None, args, device, args.seed, args.epochs, augment=False, on_epoch=report)
    (out / "overfit.json").write_text(json.dumps({"config": {**vars(args), "device": str(device)},
                                                  "history": history}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cubes", required=True)
    parser.add_argument("--cohort", default="data/cohort/nsclc_radiogenomics.csv")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--target", default="histology", choices=["histology", "egfr", "kras"])
    parser.add_argument("--subset", type=int, default=0, help="use a stratified subset of N patients (small data)")
    parser.add_argument("--pretrained", default="medicalnet", choices=["medicalnet", "none"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="", help="where to save the final model (default: <out>/final_model.pt)")
    parser.add_argument("--overfit", type=int, default=0,
                        help="sanity check: train on N patients (no augmentation, no early stopping) and report "
                             "the training loss/AUC per epoch; a working setup drives the loss towards 0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})
    df = labelled_patients(cohort, Path(args.cubes).expanduser(), args.target)
    if args.overfit:
        overfit_check(df, args, device, out)
        return
    if args.subset and args.subset < len(df):
        df, _ = train_test_split(df, train_size=args.subset, stratify=df["y"], random_state=args.seed)
    df = df.merge(load_splits(args.splits), on="patient_id").reset_index(drop=True)
    print(f"{len(df)} patients, {int(df['y'].sum())} positive; device {device}; pretrained {args.pretrained}")

    rows, histories, best_epochs = [], {}, []
    for fold in sorted(df["fold"].unique()):
        t0 = time.time()
        train_all, test = df[df["fold"] != fold], df[df["fold"] == fold]
        tr, val = train_test_split(train_all, test_size=args.val_fraction, stratify=train_all["y"],
                                   random_state=args.seed + int(fold))
        print(f"fold {fold}: train {len(tr)} / val {len(val)} / test {len(test)} ({int(test['y'].sum())} positive)")
        model, history, best_epoch = train(tr, val, args, device, args.seed + int(fold), args.epochs)
        probs = predict(model, make_loader(test, args, False, False, 0), device)
        # pos_weight balances the classes in the loss, so the matching decision threshold is 0.5
        # (not the prevalence, which would call almost every patient positive)
        rows.append(pd.DataFrame({"patient_id": test["patient_id"], "fold": fold, "y": test["y"].astype(int),
                                  "prob": probs, "threshold": 0.5}))
        histories[int(fold)] = history
        best_epochs.append(best_epoch)
        print(f"fold {fold} done in {(time.time() - t0) / 60:.1f} min, best epoch {best_epoch}")

    oof = pd.concat(rows, ignore_index=True)
    oof.to_csv(out / "oof.csv", index=False)
    pooled = binary_metrics(oof["y"], oof["prob"], oof["threshold"])
    lo, hi = bootstrap_ci(oof["y"].to_numpy(), oof["prob"].to_numpy(), n_resamples=args.bootstrap, seed=args.seed)
    per_fold = {int(f): binary_metrics(g["y"], g["prob"])["auc"] for f, g in oof.groupby("fold") if g["y"].nunique() == 2}
    metrics = {
        **pooled, "auc_ci95": [lo, hi], "auc_per_fold": per_fold,
        "patients": len(oof), "positives": int(oof["y"].sum()),
        "per_fold_counts": oof.groupby("fold")["y"].agg(patients="count", positives="sum").reset_index().to_dict("records"),
        "best_epochs": best_epochs,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({**vars(args), "device": str(device)}, indent=2), encoding="utf-8")
    print(f"\n{args.target}: AUC {pooled['auc']:.3f} (95% CI {lo:.3f}–{hi:.3f}), F1 {pooled['f1']:.3f}, "
          f"sens {pooled['sensitivity']:.3f}, spec {pooled['specificity']:.3f}, "
          f"{len(oof)} patients / {int(oof['y'].sum())} positive")

    final_epochs = int(np.median(best_epochs))
    print(f"\nFinal model: all {len(df)} patients, {final_epochs} epochs")
    model, _, _ = train(df, None, args, device, args.seed, final_epochs)
    ckpt = Path(args.checkpoint) if args.checkpoint else out / "final_model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "target": args.target, "in_channels": 1,
                "pretrained": args.pretrained, "cv_auc": [pooled["auc"], lo, hi],
                "patients": len(df), "positives": int(df["y"].sum()), "epochs": final_epochs}, ckpt)
    print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
