"""Minimal, explicit fine-tuning loop.

Written by hand (rather than ``Seq2SeqTrainer``) because Indic-Transcribe is a
``trust_remote_code`` model with a custom prompt format and feature extractor:
owning the loop makes it obvious exactly what the model sees and lets us use a
duration-bucketed sampler and WER-based model selection.
"""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import ExperimentConfig
from .data import DurationBucketSampler, ManifestDataset
from .metrics import compute_metrics
from .model import AsrAdapter

log = logging.getLogger(__name__)


def _autocast(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _lr_lambda(warmup: int, total: int):
    """Linear warmup then cosine decay to 10% of peak."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def make_loader(ds: ManifestDataset, adapter: AsrAdapter, cfg: ExperimentConfig,
                shuffle: bool) -> DataLoader:
    sampler = DurationBucketSampler(
        [it["duration"] for it in ds.items],
        max_batch_seconds=cfg.data.max_batch_seconds,
        max_batch_size=cfg.data.max_batch_size,
        shuffle=shuffle, seed=cfg.train.seed,
    )
    return DataLoader(ds, batch_sampler=sampler, collate_fn=adapter.collate,
                      num_workers=cfg.data.num_workers, pin_memory=True,
                      persistent_workers=cfg.data.num_workers > 0)


@torch.no_grad()
def evaluate(adapter: AsrAdapter, loader: DataLoader, cfg: ExperimentConfig,
             predictions_path: Path | None = None) -> dict:
    adapter.model.eval()
    refs, hyps, ids, loss_sum, n_batches = [], [], [], 0.0, 0
    for batch in loader:
        batch = adapter.to_device(batch)
        with _autocast(cfg.train.precision):
            loss_sum += adapter.loss(batch).item()
            n_batches += 1
            hyps += adapter.transcribe(batch, num_beams=cfg.train.num_beams,
                                       max_new_tokens=cfg.train.max_new_tokens)
        refs += batch["texts"]
        ids += batch["utt_ids"]
    metrics = compute_metrics(refs, hyps)
    metrics["loss"] = loss_sum / max(1, n_batches)
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            for i, r, h in zip(ids, refs, hyps):
                f.write(json.dumps({"utt_id": i, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")
    adapter.train_mode()
    return metrics


def train(cfg: ExperimentConfig) -> dict:
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.manual_seed(cfg.train.seed)

    adapter = AsrAdapter.from_config(cfg)
    train_ds = ManifestDataset(cfg.data.train_manifest)
    dev_ds = ManifestDataset(cfg.data.dev_manifest, max_items=cfg.data.dev_max_items)
    train_loader = make_loader(train_ds, adapter, cfg, shuffle=True)
    dev_loader = make_loader(dev_ds, adapter, cfg, shuffle=False)
    log.info("train utts=%d (%.1f h) batches/epoch=%d | dev utts=%d", len(train_ds),
             sum(it["duration"] for it in train_ds.items) / 3600, len(train_loader), len(dev_ds))

    groups = adapter.param_groups(cfg.train.lr, cfg.train.encoder_lr_scale, cfg.train.weight_decay)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg.train.warmup_steps, cfg.train.max_steps))
    scaler = torch.amp.GradScaler(enabled=cfg.train.precision == "fp16")
    tb = SummaryWriter(out / "tb")
    history = out / "history.jsonl"

    def log_row(row: dict) -> None:
        with history.open("a") as f:
            f.write(json.dumps(row) + "\n")
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != "step":
                tb.add_scalar(k, v, row["step"])

    # Step-0 evaluation = zero-shot baseline on the same dev set.
    best = evaluate(adapter, dev_loader, cfg)
    log.info("step 0 (zero-shot) dev: %s", best)
    log_row({"step": 0, **{f"dev/{k}": v for k, v in best.items()}})
    best_wer, bad_evals = best["wer"], 0

    step, epoch, t0 = 0, 0, time.time()
    adapter.train_mode()
    running = 0.0
    done = False
    while not done:
        train_loader.batch_sampler.set_epoch(epoch)
        for micro, batch in enumerate(train_loader):
            batch = adapter.to_device(batch)
            with _autocast(cfg.train.precision):
                loss = adapter.loss(batch) / cfg.train.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
            if (micro + 1) % cfg.train.grad_accum:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), cfg.train.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % cfg.train.log_every == 0:
                row = {"step": step, "epoch": epoch, "train/loss": running / cfg.train.log_every,
                       "train/grad_norm": float(gnorm), "train/lr": sched.get_last_lr()[0],
                       "train/elapsed_min": (time.time() - t0) / 60,
                       "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
                log_row(row)
                running = 0.0

            if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps:
                m = evaluate(adapter, dev_loader, cfg)
                log.info("step %d dev: %s", step, m)
                log_row({"step": step, **{f"dev/{k}": v for k, v in m.items()}})
                if m["wer"] < best_wer:
                    best_wer, bad_evals = m["wer"], 0
                    if cfg.train.save_best:
                        adapter.save(out / "best")
                        (out / "best" / "dev_metrics.json").write_text(json.dumps({"step": step, **m}, indent=2))
                else:
                    bad_evals += 1
                    if cfg.train.early_stopping_patience and bad_evals >= cfg.train.early_stopping_patience:
                        log.info("early stopping at step %d (best dev WER %.2f)", step, best_wer)
                        done = True
                        break
            if step >= cfg.train.max_steps:
                done = True
                break
        epoch += 1

    adapter.save(out / "last")
    tb.close()
    summary = {"best_dev_wer": best_wer, "steps": step, "epochs": epoch,
               "train_minutes": (time.time() - t0) / 60}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
