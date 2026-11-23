"""How much do Model B's Kaggle test images overlap its training images?

The Kaggle "Chest CT-Scan images" set has no patient IDs, so slices of one
patient can sit in train/ and test/. This script measures that overlap:

1. every image gets a 256-bit difference hash (16×16 dHash; near-identical
   slices get nearly identical hashes), plus an exact-byte check (MD5);
2. for each test image, the Hamming distance to its nearest training image of
   the SAME class and of the OTHER classes is computed;
3. an other-class neighbour is certainly another patient (one histology per
   patient), so the 1st percentile of the other-class distances is taken as
   "closer than different patients get" — test images whose same-class
   neighbour is below it are flagged as probably seen in training;
4. with --model, Model B's test accuracy is compared on flagged vs. unflagged images.

Usage (from the repository root):
    python scripts/check_kaggle_leakage.py --data ~/pulmognosis_data/raw/kaggle/Data \
        --model runs/model_b_cnn2d/final_model.pt --out runs/model_b_cnn2d/leakage.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from train_cnn2d import CLASSES, list_images  # same class matching as Model B

HASH_SIZE = 16  # 16×16 = 256 bits


def dhash(path: Path, size: int = HASH_SIZE) -> np.ndarray:
    """Difference hash: grayscale, resize to (size+1)×size, compare horizontally adjacent pixels."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).ravel()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def nearest_distances(test_hashes: np.ndarray, train_hashes: np.ndarray) -> np.ndarray:
    """Hamming distance from each test hash to its nearest training hash (matrix of bools in, ints out)."""
    t, r = test_hashes.astype(np.int32), train_hashes.astype(np.int32)
    # Hamming = bits differing = |t| + |r| - 2·(t·r) for 0/1 vectors
    d = t.sum(1)[:, None] + r.sum(1)[None, :] - 2 * (t @ r.T)
    return d.min(axis=1)


def model_correct(model_path: Path, items: list[tuple[Path, int]]) -> np.ndarray:
    """Whether Model B (the saved final model) classifies each test image correctly."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    from train_cnn2d import ImageFolderByPrefix, predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
    model.to(device)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    probs, labels = predict(model, DataLoader(ImageFolderByPrefix(items, tf), 32), device)
    return probs.argmax(1) == labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder containing train/, valid/, test/")
    parser.add_argument("--model", default="", help="Model B checkpoint (final_model.pt) to split accuracy by overlap")
    parser.add_argument("--percentile", type=float, default=1.0, help="of other-class distances, used as the threshold")
    parser.add_argument("--out", required=True, help="JSON file for the summary")
    args = parser.parse_args()

    data = Path(args.data).expanduser()
    train, test = list_images(data / "train"), list_images(data / "test")
    h_train = np.stack([dhash(p) for p, _ in train])
    h_test = np.stack([dhash(p) for p, _ in test])
    y_train, y_test = np.array([y for _, y in train]), np.array([y for _, y in test])

    train_md5 = {md5(p) for p, _ in train}
    exact = np.array([md5(p) in train_md5 for p, _ in test])

    d_same = np.empty(len(test), dtype=int)
    d_other = np.empty(len(test), dtype=int)
    for c in range(len(CLASSES)):
        idx = y_test == c
        if idx.any():
            d_same[idx] = nearest_distances(h_test[idx], h_train[y_train == c])
            d_other[idx] = nearest_distances(h_test[idx], h_train[y_train != c])
    threshold = float(np.percentile(d_other, args.percentile))
    flagged = (d_same < threshold) | exact

    summary = {
        "train_images": len(train), "test_images": len(test), "hash_bits": HASH_SIZE * HASH_SIZE,
        "exact_duplicates_in_train": int(exact.sum()),
        "threshold_bits": threshold,
        "threshold_rule": f"{args.percentile}th percentile of each test image's nearest OTHER-class training image",
        "flagged_test_images": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "per_class": {
            CLASSES[c]: {"test": int((y_test == c).sum()), "flagged": int(flagged[y_test == c].sum()),
                         "median_nearest_same_class": float(np.median(d_same[y_test == c])),
                         "median_nearest_other_class": float(np.median(d_other[y_test == c]))}
            for c in range(len(CLASSES)) if (y_test == c).any()
        },
        "median_nearest_same_class": float(np.median(d_same)),
        "median_nearest_other_class": float(np.median(d_other)),
    }
    if args.model:
        correct = model_correct(Path(args.model), test)
        summary["model_b_accuracy"] = {
            "all": float(correct.mean()),
            "flagged": float(correct[flagged].mean()) if flagged.any() else None,
            "not_flagged": float(correct[~flagged].mean()) if (~flagged).any() else None,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
