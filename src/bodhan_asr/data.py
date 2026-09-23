"""Dataset preparation and loading.

Raw SPRING-INX parquet shards are converted once into 16 kHz mono FLAC files plus
JSON-lines manifests in the NeMo format::

    {"audio_filepath": ..., "duration": ..., "text": ..., "source_lang": "mr",
     "target_lang": "mr", "pnc": "yes", "utt_id": ...}

Using NeMo-style manifests keeps the prepared data usable by both the
transformers training loop in this repo and NeMo's ``speech_to_text_aed.py``.
"""

from __future__ import annotations

import io
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .text import clean_target

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000


@dataclass
class FilterConfig:
    min_duration: float = 0.5
    max_duration: float = 30.0  # Canary's positional limit is 40 s; keep headroom.
    max_chars_per_sec: float = 25.0  # catches mis-segmented/misaligned utterances
    min_chars: int = 2


def _decode(audio_bytes: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # to mono
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav


def prepare_split(
    shards: list[Path],
    out_dir: Path,
    split: str,
    filt: FilterConfig,
    lang: str = "mr",
    max_items: int | None = None,
    seed: int = 0,
) -> Path:
    """Decode parquet shards -> FLAC + manifest. Returns the manifest path."""
    audio_dir = out_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=["utterance_id", "text", "audio"])
        rows.extend(table.to_pylist())
    if max_items is not None and len(rows) > max_items:
        rows = random.Random(seed).sample(rows, max_items)

    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    manifest = out_dir / f"{split}.jsonl"
    total_sec = 0.0
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            uid = row["utterance_id"]
            if uid in seen:
                reasons["duplicate_id"] += 1
                continue
            seen.add(uid)
            text = clean_target(row["text"] or "")
            if len(text) < filt.min_chars:
                reasons["empty_text"] += 1
                continue
            try:
                wav = _decode(row["audio"]["bytes"])
            except Exception:  # corrupt audio in the source corpus
                reasons["decode_error"] += 1
                continue
            dur = len(wav) / SAMPLE_RATE
            if dur < filt.min_duration:
                reasons["too_short"] += 1
                continue
            if dur > filt.max_duration:
                reasons["too_long"] += 1
                continue
            if len(text) / dur > filt.max_chars_per_sec:
                reasons["chars_per_sec"] += 1
                continue
            path = audio_dir / f"{uid}.flac"
            sf.write(path, wav, SAMPLE_RATE)
            total_sec += dur
            reasons["kept"] += 1
            f.write(json.dumps({
                "audio_filepath": str(path), "duration": round(dur, 3), "text": text,
                "source_lang": lang, "target_lang": lang, "pnc": "yes", "utt_id": uid,
            }, ensure_ascii=False) + "\n")
    log.info("%s: %s | %.2f h kept", split, dict(reasons), total_sec / 3600)
    return manifest


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ManifestDataset(Dataset):
    """Returns ``{"audio": float32 np.ndarray @16k, "text": str, "utt_id": str}``."""

    def __init__(self, manifest: str | Path, max_items: int | None = None):
        self.items = read_manifest(manifest)
        if max_items is not None:
            self.items = self.items[:max_items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        wav, _ = sf.read(it["audio_filepath"], dtype="float32")
        return {"audio": wav, "text": it["text"], "utt_id": it["utt_id"],
                "duration": it["duration"]}


class DurationBucketSampler(torch.utils.data.Sampler[list[int]]):
    """Batches utterances of similar length up to ``max_batch_seconds`` of audio.

    Cuts padding waste substantially vs. random batching (conversational speech
    ranges from <1 s to 30 s), which matters for a 1.2B-param encoder.
    """

    def __init__(self, durations: list[float], max_batch_seconds: float,
                 max_batch_size: int = 64, shuffle: bool = True, seed: int = 0):
        self.durations = durations
        self.max_batch_seconds = max_batch_seconds
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._batches = self._make_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches, cur, cur_max = [], [], 0.0
        for i in idx:
            d = self.durations[i]
            # padded cost of the batch = batch_size * longest item
            if cur and (max(cur_max, d) * (len(cur) + 1) > self.max_batch_seconds
                        or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur, cur_max = [], 0.0
            cur.append(i)
            cur_max = max(cur_max, d)
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
