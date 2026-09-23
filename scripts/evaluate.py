"""Score a checkpoint (or the untouched base model) on a manifest.

    # zero-shot baseline
    python scripts/evaluate.py --config configs/mr_full.yaml --split test --tag base
    # fine-tuned
    python scripts/evaluate.py --config configs/mr_full.yaml --split test \
        --checkpoint /content/runs/mr_full/best --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bodhan_asr.config import ExperimentConfig  # noqa: E402
from bodhan_asr.data import ManifestDataset  # noqa: E402
from bodhan_asr.model import AsrAdapter  # noqa: E402
from bodhan_asr.trainer import evaluate, make_loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--checkpoint", default=None, help="dir saved by AsrAdapter.save")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = ExperimentConfig.load(args.config, args.overrides)
    adapter = AsrAdapter.from_config(cfg, checkpoint=args.checkpoint)
    manifest = cfg.data.test_manifest if args.split == "test" else cfg.data.dev_manifest
    loader = make_loader(ManifestDataset(manifest), adapter, cfg, shuffle=False)

    out = Path(args.out_dir or Path(cfg.train.output_dir) / "eval")
    metrics = evaluate(adapter, loader, cfg, predictions_path=out / f"{args.split}_{args.tag}.jsonl")
    metrics.update(split=args.split, tag=args.tag, checkpoint=args.checkpoint)
    (out / f"{args.split}_{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
