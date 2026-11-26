"""Model B: 2D CNN (ImageNet-pretrained ResNet-18, fine-tuned) on the Kaggle "Chest CT-Scan images" set.

Classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal.
Uses the dataset's own train / valid / test folders: the images have no patient
IDs, so a patient-level split is impossible and slices of the same patient may
appear in several folders (possible leakage) — report the results with that caveat.
The class folder names differ between train/ and test/, so classes are matched
by name prefix.

Usage (from the repository root, on a GPU machine):
    python scripts/train_cnn2d.py --data ~/pulmognosis_data/raw/kaggle/Data --out runs/model_b_cnn2d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ("adenocarcinoma", "large_cell", "squamous", "normal")
_PREFIX = {"adeno": "adenocarcinoma", "large": "large_cell", "squamous": "squamous", "normal": "normal"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}


def class_of(folder_name: str) -> str | None:
    name = folder_name.lower()
    return next((cls for prefix, cls in _PREFIX.items() if name.startswith(prefix)), None)


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        cls = class_of(folder.name)
        if cls is None:
            print(f"  skipping unknown folder {folder.name}")
            continue
        items += [(f, CLASSES.index(cls)) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXT]
    return items


class ImageFolderByPrefix(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def summarise(probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_auc_ovr": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
    }


def bootstrap(probs, labels, key, n=2000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < len(CLASSES):
            continue
        stats.append(summarise(probs[idx], labels[idx])[key])
    return [float(v) for v in np.percentile(stats, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder containing train/, valid/, test/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, out = Path(args.data).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    splits = {s: list_images(data / s) for s in ("train", "valid", "test")}
    for s, items in splits.items():
        counts = np.bincount([y for _, y in items], minlength=len(CLASSES))
        print(f"{s}: {len(items)} images, per class {dict(zip(CLASSES, counts.tolist()))}")

    loaders = {
        "train": DataLoader(ImageFolderByPrefix(splits["train"], train_tf), args.batch_size, shuffle=True, num_workers=args.workers),
        "valid": DataLoader(ImageFolderByPrefix(splits["valid"], eval_tf), args.batch_size, num_workers=args.workers),
        "test": DataLoader(ImageFolderByPrefix(splits["test"], eval_tf), args.batch_size, num_workers=args.workers),
    }

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        val = summarise(*predict(model, loaders["valid"], device))
        history.append({"epoch": epoch, "train_loss": total / len(splits["train"]), **val})
        print(f"epoch {epoch:2d}  train_loss {history[-1]['train_loss']:.4f}  valid bal.acc {val['balanced_accuracy']:.3f}")
        if val["balanced_accuracy"] > best_acc:  # model selection on valid only; test is looked at once, at the end
            best_acc, best_state = val["balanced_accuracy"], {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    probs, labels = predict(model, loaders["test"], device)
    test = summarise(probs, labels)
    test["balanced_accuracy_ci95"] = bootstrap(probs, labels, "balanced_accuracy", seed=args.seed)
    test["macro_auc_ci95"] = bootstrap(probs, labels, "macro_auc_ovr", seed=args.seed)
    test["images"] = int(len(labels))
    print("\nTEST:", json.dumps(test, indent=2))

    (out / "metrics.json").write_text(json.dumps({"test": test, "best_valid_balanced_accuracy": best_acc}, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({**vars(args), "device": str(device), "classes": CLASSES}, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, out / "final_model.pt")
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
