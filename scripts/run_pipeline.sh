#!/usr/bin/env bash
# End-to-end, resumable pipeline. Every stage copies its outputs to $DRIVE_DIR as soon
# as it finishes, so a Colab disconnect / VM recycle costs at most the current stage.
# Re-running the script skips stages whose "done" marker already exists on Drive.
#
#   HF_TOKEN=... DRIVE_DIR=/content/drive/MyDrive/bodhan-asr-marathi bash scripts/run_pipeline.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/bodhan-asr-marathi}"
DATA=/content/data
RUNS=/content/runs
MODEL_DIR=/content/models/indic-transcribe-core
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
RUN_LORA="${RUN_LORA:-1}"
mkdir -p "$DRIVE_DIR/markers" "$DRIVE_DIR/runs" "$RUNS"
cd "$REPO"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
done_() { [ -f "$DRIVE_DIR/markers/$1" ]; }
mark() { date > "$DRIVE_DIR/markers/$1"; }
# small artefacts (logs, metrics, predictions, tensorboard) -> Drive; no weights
sync_small() { rsync -a --exclude 'best/' --exclude 'last/' "$RUNS/$1" "$DRIVE_DIR/runs/"; }
# restore a previous stage's outputs from Drive after a VM recycle
restore() { [ -d "$DRIVE_DIR/runs/$1" ] && rsync -a "$DRIVE_DIR/runs/$1" "$RUNS/" || true; }

log "installing dependencies"
pip -q install -r requirements.txt 2>&1 | tail -1
# Colab preinstalls torchao 0.10, which makes recent peft refuse to import
# ("only versions above 0.16.0 are supported"). Nothing here uses torchao.
pip -q uninstall -y torchao 2>/dev/null || true

log "model snapshot"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("bodhan-ai/indic-transcribe-core", ignore_patterns=["nemo/*.nemo", "*.png"],
                  local_dir="$MODEL_DIR")
EOF

if [ ! -f "$DATA/prepared/test.jsonl" ]; then
  log "data download + preparation ($TRAIN_SHARDS train shards)"
  python scripts/prepare_data.py --download --raw_dir "$DATA/raw" --out_dir "$DATA/prepared" \
      --train_shards "$TRAIN_SHARDS" --dev_items 1000 2>&1 | grep -v Warning | tee "$RUNS/prepare_data.log"
  cp "$RUNS/prepare_data.log" "$DRIVE_DIR/"
  cp "$DATA/prepared/"{train,dev,test}.jsonl "$DRIVE_DIR/" 2>/dev/null || true
fi

log "smoke test"
python scripts/smoke_test.py configs/mr_full.yaml 2>&1 | grep -v Warning | tee "$DRIVE_DIR/smoke_test.log"
grep -q "SMOKE TEST OK" "$DRIVE_DIR/smoke_test.log"

if ! done_ baseline; then
  log "zero-shot baseline on test (mixed + native prompt)"
  for mode in mixed native; do
    python scripts/evaluate.py --config configs/mr_full.yaml --split test --tag "base_$mode" \
        --out_dir "$RUNS/baseline" model.prompt_mode=$mode 2>&1 | grep -v Warning | tail -12
  done
  sync_small baseline; mark baseline
fi

if ! done_ full_train; then
  log "full fine-tuning"
  python scripts/train.py --config configs/mr_full.yaml 2>&1 | grep -v Warning > "$RUNS/mr_full.stdout" || true
  mkdir -p "$RUNS/mr_full" && mv "$RUNS/mr_full.stdout" "$RUNS/mr_full/stdout.txt"
  test -f "$RUNS/mr_full/best/model.safetensors"
  sync_small mr_full
  log "copying best full checkpoint to Drive"
  rsync -a "$RUNS/mr_full/best" "$DRIVE_DIR/runs/mr_full/"
  mark full_train
fi

if ! done_ full_eval; then
  log "test evaluation of the fine-tuned model"
  [ -d "$RUNS/mr_full/best" ] || rsync -a "$DRIVE_DIR/runs/mr_full/best" "$RUNS/mr_full/"
  python scripts/evaluate.py --config configs/mr_full.yaml --split test --tag finetuned \
      --checkpoint "$RUNS/mr_full/best" 2>&1 | grep -v Warning | tail -12
  sync_small mr_full; mark full_eval
fi

if [ "$RUN_LORA" = "1" ] && ! done_ lora; then
  log "LoRA comparison run"
  python scripts/train.py --config configs/mr_lora.yaml 2>&1 | grep -v Warning > "$RUNS/mr_lora.stdout" || true
  mkdir -p "$RUNS/mr_lora" && mv "$RUNS/mr_lora.stdout" "$RUNS/mr_lora/stdout.txt"
  if [ -f "$RUNS/mr_lora/best/adapter.safetensors" ]; then
    python scripts/evaluate.py --config configs/mr_lora.yaml --split test --tag finetuned_lora \
        --checkpoint "$RUNS/mr_lora/best" 2>&1 | grep -v Warning | tail -12
  fi
  sync_small mr_lora
  rsync -a "$RUNS/mr_lora/best" "$DRIVE_DIR/runs/mr_lora/" 2>/dev/null || true  # adapter is ~100 MB
  mark lora
fi

log "report"
for r in baseline mr_full mr_lora; do restore "$r"; done
RUN_DIRS="$RUNS/mr_full"; [ -f "$RUNS/mr_lora/history.jsonl" ] && RUN_DIRS="$RUN_DIRS $RUNS/mr_lora"
python scripts/report.py --runs $RUN_DIRS --baseline "$RUNS/baseline" --out "$DRIVE_DIR/docs"
log "PIPELINE DONE -> $DRIVE_DIR"
