"""Build the NSCLC-Radiogenomics cohort table from the TCIA metadata.

Usage (from the repository root):
    python scripts/build_cohort.py [--metadata data/metadata] [--out data/cohort/nsclc_radiogenomics.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pulmognosis.data.cohort import build_cohort, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata")
    parser.add_argument("--out", default="data/cohort/nsclc_radiogenomics.csv")
    args = parser.parse_args()

    cohort = build_cohort(args.metadata)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out, index=False)
    print(f"Wrote {len(cohort)} patients to {out}")
    print(json.dumps(summarise(cohort), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
