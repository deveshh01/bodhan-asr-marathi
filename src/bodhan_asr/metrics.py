"""WER / CER computation with the normalisation from :mod:`bodhan_asr.text`."""

from __future__ import annotations

import jiwer

from .text import normalize_for_wer


def compute_metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Corpus-level WER and CER (percent) on normalised text.

    Empty references are dropped (WER is undefined for them); empty hypotheses
    are kept - they are genuine deletions.
    """
    pairs = [(normalize_for_wer(r), normalize_for_wer(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    r, h = map(list, zip(*pairs))
    return {
        "wer": 100 * jiwer.wer(r, h),
        "cer": 100 * jiwer.cer(r, h),
        "n": len(r),
    }
