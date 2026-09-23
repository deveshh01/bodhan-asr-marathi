"""Fine-tune Indic-Transcribe on Marathi.

    python scripts/train.py --config configs/mr_full.yaml [train.max_steps=500 ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bodhan_asr.config import ExperimentConfig  # noqa: E402
from bodhan_asr.trainer import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="section.key=value")
    args = ap.parse_args()
    cfg = ExperimentConfig.load(args.config, args.overrides)
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(Path(cfg.train.output_dir) / "train.log")],
    )
    print(json.dumps(train(cfg), indent=2))


if __name__ == "__main__":
    main()
