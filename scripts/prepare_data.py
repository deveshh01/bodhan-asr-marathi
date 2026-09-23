"""Download (optional) and prepare SPRING-INX Marathi R2 into FLAC + manifests.

Example::

    python scripts/prepare_data.py --raw_dir /content/data/raw --out_dir /content/data/prepared \
        --train_shards 10 --dev_items 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bodhan_asr.data import FilterConfig, prepare_split  # noqa: E402

REPO = "SPRINGLab/SPRING_INX_Marathi_R2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--train_shards", type=int, default=10, help="of 47 available")
    ap.add_argument("--dev_items", type=int, default=1000,
                    help="random subset of validation shard 0 used for model selection")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    names = {
        "train": [f"data/train-{i:05d}-of-00047.parquet" for i in range(args.train_shards)],
        "dev": ["data/validation-00000-of-00006.parquet"],
        "test": [f"data/test-{i:05d}-of-00002.parquet" for i in range(2)],
    }
    if args.download:
        from huggingface_hub import hf_hub_download
        for files in names.values():
            for n in files:
                hf_hub_download(REPO, n, repo_type="dataset", local_dir=args.raw_dir)

    filt = FilterConfig()
    for split, files in names.items():
        shards = [args.raw_dir / n for n in files]
        # The official test split is never filtered by duration/text heuristics
        # beyond decodability, so reported test numbers stay comparable.
        split_filt = FilterConfig(max_duration=40.0, max_chars_per_sec=1e9) if split == "test" else filt
        prepare_split(shards, args.out_dir, split, split_filt,
                      max_items=args.dev_items if split == "dev" else None)


if __name__ == "__main__":
    main()
