"""Aggregate a run into plots + a markdown results table + error examples.

    python scripts/report.py --runs /content/runs/mr_full /content/runs/mr_lora \
        --baseline /content/runs/baseline --out docs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bodhan_asr.text import normalize_for_wer  # noqa: E402


def load_history(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line]


def plot_runs(runs: list[Path], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run in runs:
        h = load_history(run)
        tr = [r for r in h if "train/loss" in r]
        dv = [r for r in h if "dev/wer" in r]
        axes[0].plot([r["step"] for r in tr], [r["train/loss"] for r in tr], label=run.name, alpha=0.8)
        axes[1].plot([r["step"] for r in dv], [r["dev/loss"] for r in dv], marker="o", label=run.name)
        axes[2].plot([r["step"] for r in dv], [r["dev/wer"] for r in dv], marker="o", label=f"{run.name} WER")
        axes[2].plot([r["step"] for r in dv], [r["dev/cer"] for r in dv], marker="x", ls="--", label=f"{run.name} CER")
    for ax, title in zip(axes, ["train loss", "dev loss", "dev WER / CER (%)"]):
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=120)


def metrics_table(files: list[Path]) -> str:
    rows = ["| model | split | WER % | CER % | utts |", "|---|---|---|---|---|"]
    for f in files:
        m = json.loads(f.read_text())
        rows.append(f"| {m['tag']} | {m['split']} | {m['wer']:.2f} | {m['cer']:.2f} | {m['n']} |")
    return "\n".join(rows)


def examples(base: Path, tuned: Path, k: int = 8) -> str:
    """Utterances where fine-tuning changed the output the most (either way)."""
    import jiwer

    b = {r["utt_id"]: r for r in map(json.loads, base.read_text().splitlines())}
    t = {r["utt_id"]: r for r in map(json.loads, tuned.read_text().splitlines())}
    scored = []
    for uid, r in t.items():
        ref = normalize_for_wer(r["ref"])
        if uid not in b or not ref:
            continue
        wb = jiwer.wer(ref, normalize_for_wer(b[uid]["hyp"]))
        wt = jiwer.wer(ref, normalize_for_wer(r["hyp"]))
        scored.append((wb - wt, uid, r["ref"], b[uid]["hyp"], r["hyp"], wb, wt))
    scored.sort()
    pick = scored[-k // 2:][::-1] + scored[: k // 2]  # biggest wins, then biggest regressions
    out = []
    for gain, uid, ref, hb, ht, wb, wt in pick:
        out.append(f"**{uid}** (utt WER {100 * wb:.0f}% -> {100 * wt:.0f}%)\n\n"
                   f"- REF: {ref}\n- BASE: {hb}\n- FT: {ht}\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_runs(args.runs, args.out)
    files = sorted(args.baseline.glob("*_metrics.json"))
    for run in args.runs:
        files += sorted((run / "eval").glob("*_metrics.json"))
    md = ["## Results\n", metrics_table(files), ""]
    base_pred = args.baseline / "test_base_mixed.jsonl"
    tuned_pred = args.runs[0] / "eval" / "test_finetuned.jsonl"
    if base_pred.exists() and tuned_pred.exists():
        md += ["## Largest changes on test (baseline -> fine-tuned)\n", examples(base_pred, tuned_pred)]
    (args.out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
