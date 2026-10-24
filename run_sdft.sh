#!/usr/bin/env bash
set -euo pipefail

# Run from the repository with your environment already activated.
MODE="${MODE:-train}"              # train or eval
MODEL_SIZE="${MODEL_SIZE:-4b}"      # 4b full FT or 9b LoRA
TEACHER_KIND="${TEACHER_KIND:-ema}" # ema or base
MASTER_PORT="${MASTER_PORT:-29500}"
case "$TEACHER_KIND" in ema|base) ;; *) echo 'TEACHER_KIND must be ema or base' >&2; exit 1 ;; esac

LORA_ARGS=()
case "$MODEL_SIZE" in
  4b) MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; LR="${LR:-1e-5}"; EVAL_BS="${EVAL_BS:-8}" ;;
  9b)
    MODEL="${MODEL:-Qwen/Qwen3.5-9B}"; LR="${LR:-1e-4}"; EVAL_BS="${EVAL_BS:-4}"
    LORA_ARGS=(--use_peft --lora_r "${LORA_R:-64}" --lora_alpha "${LORA_ALPHA:-128}"
      --lora_dropout "${LORA_DROPOUT:-0.05}" --lora_target_modules "${LORA_TARGETS:-all-linear}"
      --lora_task_type CAUSAL_LM)
    ;;
  *) echo 'MODEL_SIZE must be 4b or 9b' >&2; exit 1 ;;
esac

BS="${BS:-8}"
GA="${GA:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DISTILLATION_MODE="${DISTILLATION_MODE:-full_logits}"
DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.0}"
RUN_NAME="sdft-${MODEL_SIZE}-${TEACHER_KIND}-lr-${LR}"
if [[ "$MODEL_SIZE" == 9b ]]; then
  RUN_NAME="${RUN_NAME}-r${LORA_R:-64}-alpha${LORA_ALPHA:-128}"
fi
TRAIN_OUT="${TRAIN_OUT:-${SDFT_OUTPUT_ROOT:-outputs}/${RUN_NAME}}"
TEACHER_ARGS=(--teacher_model_kind "$TEACHER_KIND")
if [[ "$TEACHER_KIND" == ema ]]; then
  TEACHER_ARGS+=(--teacher_sync_steps "${TEACHER_SYNC_STEPS:-5}"
    --teacher_update_rate "${TEACHER_UPDATE_RATE:-0.01}")
fi
DEMONSTRATION_ARGS=()
if [[ -n "${DEMONSTRATION_TEMPLATE:-}" ]]; then
  DEMONSTRATION_ARGS=(--demonstration_template "$DEMONSTRATION_TEMPLATE")
fi

EXTRA=()
case "$MODE" in
  train)
    ENTRY=tooluse_sdft_train_entry.py
    SAVE_ARGS=(--save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 1 --eval_on_save)
    REPORT_TO=tensorboard
    if [[ -n "${RESUME:-}" ]]; then
      EXTRA=(--resume_from_checkpoint "$RESUME")
    else
      EXTRA=(--no_auto_resume)
    fi
    ;;
  eval)
    : "${CKPT:?Set CKPT to the checkpoint directory}"
    ENTRY=tooluse_eval_entry.py
    TRAIN_OUT="$(dirname "$CKPT")"
    SAVE_ARGS=(--save_strategy no)
    REPORT_TO=none
    EXTRA=(--resume_from_checkpoint "$CKPT" --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS:-1024}")
    ;;
  *) echo 'MODE must be train or eval' >&2; exit 1 ;;
esac
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-checkpoint_evals/$(basename "$TRAIN_OUT")}"

accelerate launch \
  --use_deepspeed \
  --num_processes 4 \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision bf16 \
  --zero3_init_flag true \
  --deepspeed_config_file configs/deepspeed/zero3.json \
  "${ENTRY}" \
  --model_name_or_path "${MODEL}" \
  --dataset_name Ahren09/ToolAlpaca \
  "${DEMONSTRATION_ARGS[@]}" \
  --output_dir "${TRAIN_OUT}" \
  --per_device_train_batch_size "${BS}" \
  --gradient_accumulation_steps "${GA}" \
  --learning_rate "${LR}" \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_prompt_length 1024 \
  --max_completion_length 512 \
  --distillation_mode "${DISTILLATION_MODE}" \
  --distillation_alpha "${DISTILLATION_ALPHA}" \
  "${TEACHER_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  --dtype bfloat16 \
  --gradient_checkpointing \
  "${SAVE_ARGS[@]}" \
  --eval_batch_size "${EVAL_BS}" \
  --eval_artifact_dir "${EVAL_RESULTS_DIR}" \
  --dataloader_num_workers 0 \
  --no_dataloader_pin_memory \
  --eval_strategy no \
  --report_to "${REPORT_TO}" \
  "${EXTRA[@]}"
