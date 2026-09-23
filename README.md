# Fine-tuning Bodhan Indic-Transcribe (ASR) on conversational Marathi

End-to-end fine-tuning of [`bodhan-ai/indic-transcribe-core`](https://huggingface.co/bodhan-ai/indic-transcribe-core)
(1.2B-param FastConformer + Transformer AED, a Canary-1B-v2 derivative) on
[SPRING-INX Marathi R2](https://huggingface.co/datasets/SPRINGLab/SPRING_INX_Marathi_R2)
(IIT Madras SPRING Lab; conversational, code-mixed Marathi).

Everything runs from one Colab notebook (A100 80GB); the notebook's `%%writefile`
cells are a 1:1 mirror of this repository.

<!-- RESULTS -->

## Repository layout

```
configs/
  mr_full.yaml         full fine-tuning (main run)
  mr_lora.yaml         LoRA variant (parameter-efficient comparison)
scripts/
  prepare_data.py      parquet shards -> 16 kHz FLAC + NeMo-style JSONL manifests
  smoke_test.py        load, decode 6 clips, 1 optimizer step - run before any long job
  train.py             fine-tuning entry point (YAML config + key=value overrides)
  evaluate.py          WER/CER of base model or a checkpoint, writes per-utterance predictions
  report.py            training curves, results table, biggest wins/regressions
src/bodhan_asr/
  config.py            typed dataclass config
  data.py              data prep, manifest dataset, duration-bucketed batch sampler
  model.py             training adapter around the (inference-only) upstream HF port
  trainer.py           explicit training loop: bf16, warmup+cosine, WER model selection
  text.py              target cleaning vs. scoring normalisation
  metrics.py           corpus WER / CER
notebooks/
  bodhan_asr_marathi_colab.ipynb   the notebook that produced the results
```

## Reproduce

```bash
pip install -r requirements.txt
# HF token with access to the gated bodhan-ai/indic-transcribe-core repo
huggingface-cli download bodhan-ai/indic-transcribe-core --exclude "nemo/*.nemo" \
    --local-dir /content/models/indic-transcribe-core
python scripts/prepare_data.py --raw_dir /content/data/raw --out_dir /content/data/prepared --download
python scripts/smoke_test.py configs/mr_full.yaml
python scripts/train.py --config configs/mr_full.yaml
python scripts/evaluate.py --config configs/mr_full.yaml --split test --tag base_mixed \
    --out_dir /content/runs/baseline
python scripts/evaluate.py --config configs/mr_full.yaml --split test --tag finetuned \
    --checkpoint /content/runs/mr_full/best
python scripts/report.py --runs /content/runs/mr_full --baseline /content/runs/baseline --out docs
```

The saved `best/` directory is self-contained (weights + upstream code, tokenizer and
feature extractor), so it also loads with Bodhan's own wrapper:

```python
import sys; sys.path.insert(0, "runs/mr_full/best")
from indic_transcribe import IndicTranscribe
asr = IndicTranscribe.from_pretrained("runs/mr_full/best")
print(asr("clip.wav", lang="mr", mode="mixed"))
```

## Approach

### 1. Choosing the model

| model | backbone | size (bf16) | verdict |
|---|---|---|---|
| Indic-Translate | Gemma-4 E4B | ~16 GB | trainable only with QLoRA; long download; MT eval is noisy |
| Indic-Speak | Llama-3.2-3B + Vocos codec | ~7 GB | needs an audio *encoder* to tokenise targets; the repo ships only the decoder side |
| **Indic-Transcribe** | FastConformer + Transformer AED | ~2.4 GB | **chosen**: fits comfortably, clean objective (CE on text), unambiguous metrics (WER/CER) |

ASR also gives the most honest "did fine-tuning do anything" signal: a zero-shot
baseline on the same test set, measured with the same code.

### 2. Choosing the data

SPRING-INX Marathi R2 (76k train utterances, ~350 h total, CC-BY-4.0) was picked over
read-speech corpora because it is **conversational and code-mixed** - speakers switch into
English and Hindi mid-sentence - which is exactly where a general model is weakest and
where the corpus's transcription conventions (English words in Latin script, punctuation)
differ from the model's defaults. The smoke test showed this immediately:

```
REF: company च्या आर्थिक संरचनेमध्ये ...        REF: त्याच्यामधे late हौन जाते रात्री.
HYP: कंपनीच्या आर्थिक संरचनेमध्ये ...           HYP: त्याच्यामध्ये लेट होऊन जातो रात्री
```

For time/budget reasons I used **10 of 47 train shards (16,260 utterances, 45.3 h)**,
a fixed random **1,000-utterance dev subset** of the official validation split for model
selection, and the **full official test split (1,929 utterances, 5.05 h)** for reporting.

Data preparation (`scripts/prepare_data.py`):
- decode to mono 16 kHz FLAC (the corpus is already 16 kHz PCM - verified, so no
  resampling happens; this matters because the model card warns that only torchaudio's
  `sinc_interp_hann` matches training-time resampling);
- NFC-normalise text, strip annotation tags, **keep punctuation and Latin-script words**;
- filters (train/dev only): 0.5-30 s duration (model trained with `max_duration: 30`),
  >25 chars/s (misaligned segments), empty text. Dropped: 14 too long, 33 chars/s, 3 empty.
  The test split is only filtered for decodability/emptiness so numbers stay comparable;
- manifests use the NeMo format so the same data also works with NeMo's
  `speech_to_text_aed.py` if one wants the official recipe.

### 3. Making an inference-only port trainable

The released HF code is explicitly inference-only (`forward(labels=...)` raises
`NotImplementedError`). Rather than patch upstream files, `src/bodhan_asr/model.py` wraps them:

- **Loss**: teacher-forced cross-entropy over `prompt + target + <eos>`. Targets are the
  multilingual SentencePiece ids offset by 1152 (the aggregate-tokenizer layout). The fixed
  10-token canary2 prompt is **masked out of the loss**, as in NeMo.
- **Prompt / output mode**: the model has three output modes selected by prompt slots
  (`native`, `mixed`=ITN, `romanised`). SPRING-INX writes English loanwords in Latin
  script, which matches `mixed`, so both training and evaluation use the `mixed` prompt
  (`<|mr|><|mr|><|pnc|><|itn|>…`). The baseline is reported in both modes so the choice is
  visible, not hidden.
- **Features** come from the upstream `IndicCanaryFeatureExtractor` on the GPU (bit-exact
  with inference), with short clips centre-padded to 1 s exactly like the upstream wrapper.
- **SpecAugment** (2×27 freq masks, 10×5% time masks - Canary defaults): the port contains
  no dropout at all, so this is the main regulariser.
- **BatchNorm running stats frozen**: the FastConformer conv module has BatchNorm; letting it
  re-estimate statistics from small, padding-contaminated fine-tuning batches is a classic way
  to silently degrade a pretrained conformer.
- **Checkpoints** are saved as a bf16 *copy* of the fp32 master weights (2.4 GB instead of
  4.9 GB) plus the upstream code/tokenizer/feature-extractor files, so a checkpoint dir is a
  drop-in replacement for the original repo.
- Strategies: `full`, `decoder_only`, `lora` (LoRA injected with `peft.inject_adapter_in_model`
  so the upstream `generate()` signature is untouched).

### 4. Training setup

| | value | why |
|---|---|---|
| strategy | full fine-tuning | A100 80GB makes it affordable (peak ~51 GB); the model already knows Marathi, we are adapting domain + conventions |
| LR | 1e-5, 150 warmup, cosine to 10% | low LR to limit catastrophic forgetting of the other 24 languages |
| batching | duration buckets, ≤600 padded audio-seconds, ≤48 utts | conversational clips range 0.5-30 s; bucketing cuts padding waste |
| precision | bf16 autocast, fp32 master weights | A100 native; no loss scaling needed |
| optimiser | AdamW (β2=0.98), WD 0.01 (not on norms/biases/embeddings), clip 1.0 | standard for AED transformers |
| model selection | dev WER every 250 steps, early stopping patience 3 | select on the metric we report, not on loss |
| step-0 eval | zero-shot dev WER logged before any update | the baseline and the fine-tuned model share code paths |

WER/CER are computed on **normalised** text (NFC, punctuation removed, lower-cased Latin,
zero-width chars dropped) for both reference and hypothesis, so the metric measures words
rather than punctuation conventions. Predictions keep punctuation.

## Challenges and how they were handled

- **Gated model + fine-grained HF token**: the first token returned 403 on every Bodhan repo
  even after licence acceptance; fine-grained tokens need the "public gated repos" read
  permission. Diagnosed with a small script that checks each repo's `config.json`.
- **Inference-only upstream code**: implemented the loss, prompt masking, SpecAugment and BN
  freezing in a wrapper instead of forking upstream files (keeps parity with the released model).
- **Silent weight corruption trap**: upstream `_init_weights` is deliberately a no-op because
  transformers 5.x re-initialises after loading. Any wrapper that re-enabled init or cast the
  live model would corrupt weights without an error - hence the bf16 *copy* on save and a
  smoke test that decodes real audio before any long run.
- **Transcript conventions vs. model conventions**: references mix scripts and punctuation
  inconsistently; handled with the `mixed` prompt for training and normalised scoring.
- **Noisy references**: several references omit words that are clearly spoken (the model's
  zero-shot hypothesis contains them). This caps attainable WER and means small WER changes
  should be read with care; the per-utterance prediction files make this auditable.
- **Colab ergonomics**: long jobs run as background processes (logs + TensorBoard on disk)
  so the notebook stays responsive and a browser disconnect cannot kill training.

## Limitations / next steps

- Only 45 h of the ~300 h train split were used; the pipeline scales by changing `--train_shards`.
- Forgetting on other languages was not measured - a Hindi/English sanity set would be the
  first thing to add before shipping.
- Beam search, LM fusion and a NeMo-recipe comparison run were out of scope for the time box.
