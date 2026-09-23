"""Typed experiment config loaded from YAML (with CLI ``key=value`` overrides)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    train_manifest: str = "/content/data/prepared/train.jsonl"
    dev_manifest: str = "/content/data/prepared/dev.jsonl"
    test_manifest: str = "/content/data/prepared/test.jsonl"
    lang: str = "mr"
    max_batch_seconds: float = 600.0  # padded audio-seconds per micro-batch
    max_batch_size: int = 48
    num_workers: int = 8
    dev_max_items: int | None = None


@dataclass
class ModelConfig:
    name_or_path: str = "bodhan-ai/indic-transcribe-core"
    # full | decoder_only | lora
    strategy: str = "full"
    freeze_encoder_layers: int = 0  # freeze the bottom N FastConformer layers
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # None -> all attention projections
    # canary2 output mode used in the prompt: native | mixed | romanised
    prompt_mode: str = "mixed"
    spec_augment: bool = True
    label_smoothing: float = 0.0


@dataclass
class TrainConfig:
    output_dir: str = "/content/runs/mr_full"
    seed: int = 42
    lr: float = 1e-5
    encoder_lr_scale: float = 1.0  # encoder LR = lr * scale (discriminative LR)
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 3000
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    precision: str = "bf16"  # bf16 | fp16 | fp32
    eval_every: int = 250
    log_every: int = 10
    save_best: bool = True
    early_stopping_patience: int = 5  # in evals; 0 disables
    num_beams: int = 1
    max_new_tokens: int = 256


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for ov in overrides or []:
            key, val = ov.split("=", 1)
            section, name = key.split(".", 1)
            raw.setdefault(section, {})[name] = yaml.safe_load(val)
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
