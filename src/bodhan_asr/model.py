"""Training adapter around Bodhan's Indic-Transcribe (IndicCanary) HF port.

The released transformers port is *inference-only*: ``forward(labels=...)``
raises ``NotImplementedError``. This module adds what training needs without
modifying the upstream code:

* teacher-forced cross-entropy over the target tokens (the fixed 10-token
  canary2 prompt is masked out of the loss, as in NeMo's prompt-masked loss);
* SpecAugment on the log-mel features (the port has no dropout at all, so
  this is the main regulariser);
* BatchNorm running statistics frozen - the FastConformer conv module's BN
  would otherwise re-estimate stats from small, padding-contaminated batches;
* three fine-tuning strategies: ``full``, ``decoder_only`` and ``lora``.

Everything the model sees is built from the upstream tokenizer / feature
extractor so train-time and inference-time inputs are identical.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig

log = logging.getLogger(__name__)

# Files that make a checkpoint directory self-contained (loadable with the
# upstream ``IndicTranscribe.from_pretrained``).
_ASSET_GLOBS = ["*.py", "tokenizer_*.model", "tokenizer_config.json",
                "feature_extractor.safetensors", "generation_config.json", "*.md"]

# Module names inside IndicCanary attention blocks (encoder / decoder).
LORA_TARGETS = ["linear_q", "linear_k", "linear_v", "linear_out",
                "query_net", "key_net", "value_net", "out_projection"]


def _resolve_model_dir(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name_or_path, ignore_patterns=["nemo/*.nemo", "*.png"]))


def _import_upstream(model_dir: Path):
    """Import the upstream remote-code modules as flat modules (they support it)."""
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from modeling_indic_canary import IndicCanaryForConditionalGeneration
    from tokenization_indic_canary import IndicCanaryTokenizer
    return IndicCanaryForConditionalGeneration, IndicCanaryFeatureExtractor, IndicCanaryTokenizer


class SpecAugment(nn.Module):
    """Frequency + time masking on (B, n_mels, T) features (NeMo Canary defaults)."""

    def __init__(self, freq_masks=2, freq_width=27, time_masks=10, time_width=0.05):
        super().__init__()
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width = time_masks, time_width

    @torch.no_grad()
    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, n_mels, _ = feats.shape
        feats = feats.clone()
        for i in range(b):
            t = int(lengths[i])
            for _ in range(self.freq_masks):
                w = np.random.randint(0, self.freq_width + 1)
                f0 = np.random.randint(0, max(1, n_mels - w))
                feats[i, f0:f0 + w, :] = 0.0
            max_w = max(1, int(self.time_width * t))
            for _ in range(self.time_masks):
                w = np.random.randint(0, max_w + 1)
                t0 = np.random.randint(0, max(1, t - w))
                feats[i, :, t0:t0 + w] = 0.0
        return feats


class AsrAdapter:
    """Owns model + tokenizer + feature extractor and exposes the train/eval API
    used by :mod:`bodhan_asr.trainer`."""

    def __init__(self, model, fe, tokenizer, cfg: ExperimentConfig, model_dir: Path,
                 device: str = "cuda"):
        self.model, self.fe, self.tok, self.cfg = model, fe, tokenizer, cfg
        self.model_dir, self.device = model_dir, device
        self.lang = cfg.data.lang
        # "mixed" (itn) mode: SPRING-INX writes English loanwords in Latin script,
        # which is exactly what Indic-Transcribe's <|itn|> slot produces.
        self.prompt = tokenizer.encode_prompt(self.lang, itn=cfg.model.prompt_mode == "mixed",
                                              romanized=cfg.model.prompt_mode == "romanised")
        self.spec_augment = SpecAugment() if cfg.model.spec_augment else None

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_config(cls, cfg: ExperimentConfig, checkpoint: str | None = None) -> "AsrAdapter":
        model_dir = _resolve_model_dir(cfg.model.name_or_path)
        Model, FE, Tok = _import_upstream(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        strategy = cfg.model.strategy
        weights_dir = checkpoint if (checkpoint and strategy != "lora") else model_dir
        # fp32 master weights; bf16 compute comes from autocast in the trainer.
        model = Model.from_pretrained(str(weights_dir), dtype=torch.float32)
        adapter = cls(model, FE.from_pretrained(str(model_dir), device=device),
                      Tok.from_pretrained(str(model_dir)), cfg, model_dir, device)
        adapter._apply_strategy()
        if checkpoint and strategy == "lora":
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(Path(checkpoint) / "adapter.safetensors")))
            log.info("loaded LoRA adapter from %s", checkpoint)
        model.to(device)
        n_train = sum(p.numel() for p in adapter.trainable_parameters())
        n_all = sum(p.numel() for p in model.parameters())
        log.info("strategy=%s trainable params %.1fM / %.1fM (%.1f%%)", strategy,
                 n_train / 1e6, n_all / 1e6, 100 * n_train / n_all)
        return adapter

    def _apply_strategy(self) -> None:
        mc, enc = self.cfg.model, self.model.model.encoder
        if mc.strategy == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            for p in self.model.parameters():
                p.requires_grad_(False)
            # inject (not get_peft_model) so the upstream generate()/forward
            # signatures stay untouched.
            inject_adapter_in_model(LoraConfig(
                r=mc.lora_r, lora_alpha=mc.lora_alpha, lora_dropout=mc.lora_dropout,
                target_modules=mc.lora_target_modules or LORA_TARGETS), self.model)
        elif mc.strategy == "decoder_only":
            for p in enc.parameters():
                p.requires_grad_(False)
        elif mc.strategy != "full":
            raise ValueError(f"unknown strategy {mc.strategy!r}")
        # Optionally freeze the subsampling + bottom N conformer layers (acoustic
        # front-end generalises well; saves memory and limits forgetting).
        if mc.freeze_encoder_layers:
            for p in enc.pre_encode.parameters():
                p.requires_grad_(False)
            for layer in enc.layers[: mc.freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def param_groups(self, lr: float, encoder_lr_scale: float, weight_decay: float):
        enc_ids = {id(p) for p in self.model.model.encoder.parameters()}
        groups = {("enc", True): [], ("enc", False): [], ("dec", True): [], ("dec", False): []}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            decay = p.ndim >= 2 and "embedding" not in name  # no WD on norms/biases/embeddings
            groups[("enc" if id(p) in enc_ids else "dec", decay)].append(p)
        out = []
        for (part, decay), params in groups.items():
            if params:
                out.append({"params": params, "weight_decay": weight_decay if decay else 0.0,
                            "lr": lr * (encoder_lr_scale if part == "enc" else 1.0)})
        return out

    # ------------------------------------------------------------------- data
    def encode_text(self, text: str) -> list[int]:
        """Target token ids: multilingual SPM pieces offset into the aggregate vocab."""
        return [i + self.tok.spl_size for i in self.tok.multi.encode(text)]

    def collate(self, items: list[dict]) -> dict:
        min_len = self.fe.sample_rate  # upstream pads clips < 1 s to 1 s, centred
        lens = [max(len(it["audio"]), min_len) for it in items]
        audio = torch.zeros(len(items), max(lens))
        for i, it in enumerate(items):
            wav = torch.from_numpy(np.asarray(it["audio"], dtype=np.float32))
            off = (min_len - len(wav)) // 2 if len(wav) < min_len else 0
            audio[i, off:off + len(wav)] = wav
        max_tok = self.model.config.max_target_positions - len(self.prompt) - 1
        seqs = [self.prompt + self.encode_text(it["text"])[:max_tok] + [self.tok.eos_id] for it in items]
        tokens = torch.full((len(seqs), max(map(len, seqs))), self.tok.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            tokens[i, : len(s)] = torch.tensor(s)
        return {"audio": audio, "audio_lens": torch.tensor(lens), "tokens": tokens,
                "token_lens": torch.tensor([len(s) for s in seqs]),
                "texts": [it["text"] for it in items], "utt_ids": [it["utt_id"] for it in items]}

    def to_device(self, batch: dict) -> dict:
        batch = dict(batch)
        audio = batch["audio"].to(self.device, non_blocking=True)
        feats, feat_lens = self.fe(audio, batch["audio_lens"].to(self.device))
        batch["feats"], batch["feat_lens"] = feats, feat_lens
        batch["feat_mask"] = (torch.arange(feats.size(2), device=self.device)[None]
                              < feat_lens[:, None]).long()
        batch["tokens"] = batch["tokens"].to(self.device, non_blocking=True)
        batch["token_lens"] = batch["token_lens"].to(self.device)
        return batch

    # ------------------------------------------------------------- train/eval
    def train_mode(self) -> None:
        self.model.train()
        for m in self.model.modules():  # freeze BN running stats (see module doc)
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

    def loss(self, batch: dict) -> torch.Tensor:
        feats = batch["feats"]
        if self.model.training and self.spec_augment is not None:
            feats = self.spec_augment(feats, batch["feat_lens"])
        tokens = batch["tokens"]
        dec_in, target = tokens[:, :-1], tokens[:, 1:].clone()
        # target position j predicts tokens[j+1]; the first len(prompt)-1 targets
        # are prompt tokens -> masked. Padding after eos is masked too.
        pos = torch.arange(target.size(1), device=target.device)[None]
        target[(pos < len(self.prompt) - 1) | (pos >= (batch["token_lens"][:, None] - 1))] = -100
        out = self.model(input_features=feats, attention_mask=batch["feat_mask"],
                         decoder_input_ids=dec_in, use_cache=False)
        return F.cross_entropy(out.logits.float().transpose(1, 2), target, ignore_index=-100,
                               label_smoothing=self.cfg.model.label_smoothing)

    @torch.no_grad()
    def transcribe(self, batch: dict, num_beams: int = 1, max_new_tokens: int = 256) -> list[str]:
        prompt = torch.tensor([self.prompt] * batch["feats"].size(0), device=self.device)
        # max_length (not max_new_tokens): generation_config already sets
        # max_length=1024, and passing both logs a warning on every batch.
        out = self.model.generate(input_features=batch["feats"], attention_mask=batch["feat_mask"],
                                  decoder_input_ids=prompt, max_length=len(self.prompt) + max_new_tokens,
                                  num_beams=num_beams, do_sample=False)
        texts = []
        for row in out.tolist():
            ids = self.tok.strip_prompt_and_trim(row, self.prompt)
            if self.tok.eos_id in ids:  # anything after the first eos is padding
                ids = ids[: ids.index(self.tok.eos_id)]
            texts.append(self.tok.decode(ids))
        return texts

    # ------------------------------------------------------------------- save
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.model.strategy == "lora":
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file
            sd = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(self.model).items()}
            save_file(sd, str(out_dir / "adapter.safetensors"))
        else:
            # bf16 *copy* on disk (2.4 GB vs 4.9 GB; inference runs in bf16 anyway).
            # Never cast the live model: that would truncate the fp32 master
            # weights the optimizer is still updating. lm_head is tied to the
            # token embedding and re-tied on load, so it isn't stored twice.
            sd = {k: v.detach().to(torch.bfloat16) for k, v in self.model.state_dict().items()
                  if k != "lm_head.weight"}
            self.model.save_pretrained(str(out_dir), state_dict=sd, safe_serialization=True)
            for pat in _ASSET_GLOBS:  # make the dir loadable by upstream IndicTranscribe
                for f in self.model_dir.glob(pat):
                    shutil.copy2(f, out_dir / f.name)
        log.info("saved %s checkpoint -> %s", self.cfg.model.strategy, out_dir)
