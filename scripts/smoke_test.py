"""Fast end-to-end sanity check before spending GPU hours:
loads the model, decodes a few dev clips, and runs one optimisation step."""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from bodhan_asr.config import ExperimentConfig  # noqa: E402
from bodhan_asr.data import ManifestDataset  # noqa: E402
from bodhan_asr.model import AsrAdapter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
cfg = ExperimentConfig.load(sys.argv[1] if len(sys.argv) > 1 else "configs/mr_full.yaml")
print("torch", torch.__version__, "| transformers", transformers.__version__)

# 1. Native sample rate of the corpus (decides whether resampling matters at all).
raw = next(Path("/content/data/raw/data").glob("test-*.parquet"))
row = pq.read_table(raw, columns=["audio"]).slice(0, 1).to_pylist()[0]
info = sf.info(io.BytesIO(row["audio"]["bytes"]))
print("source audio:", info.samplerate, "Hz,", info.channels, "ch,", info.subtype)

# 2. Load + decode a handful of dev utterances.
adapter = AsrAdapter.from_config(cfg)
print("prompt ids:", adapter.prompt)
ds = ManifestDataset(cfg.data.dev_manifest, max_items=6)
batch = adapter.to_device(adapter.collate([ds[i] for i in range(len(ds))]))
adapter.model.eval()
with torch.autocast("cuda", dtype=torch.bfloat16):
    t = time.time()
    hyps = adapter.transcribe(batch)
    print(f"decode {len(hyps)} utts in {time.time() - t:.1f}s")
    for r, h in zip(batch["texts"], hyps):
        print("REF:", r, "\nHYP:", h, "\n")
    print("eval loss:", adapter.loss(batch).item())

# 3. One training step (checks grads flow and memory is sane).
adapter.train_mode()
opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-6)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = adapter.loss(batch)
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), 1.0)
opt.step()
print(f"train loss {loss.item():.4f} grad_norm {gn:.3f} peak_mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print("SMOKE TEST OK")
