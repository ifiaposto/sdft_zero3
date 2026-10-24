# Ready-to-run SDFT with PEFT and DeepSpeed ZeRO-3

This repository extends [Hugging Face TRL](https://github.com/huggingface/trl)’s experimental **SDFTTrainer** with ready-to-run support for practical SDFT experiments using PEFT and DeepSpeed ZeRO-3.

- **ZeRO-3-safe LoRA EMA teacher:** maintain and update the teacher adapter safely under parameter sharding.
- **EMA checkpointing and exact resume:** save and restore EMA state alongside training checkpoints.
- **Robust ZeRO-3 checkpointing:** save and restore distributed model and optimizer state.
- **Quick launchers for sdft with LoRA EMA teacher:** launch 9B LoRA training with EMA updates restricted to adapter parameters for efficiency.

> [!NOTE]
> This project extends TRL’s SDFTTrainer and, to our knowledge, provides the first experimental evaluation of LoRA adapter EMA teachers for SDFT under DeepSpeed ZeRO-3. The default launcher selects **full-vocabulary forward KL** (`DISTILLATION_MODE=full_logits`, `DISTILLATION_ALPHA=0.0`). For methodology, see [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897).

## :hammer_and_wrench: Installation


```bash
# Example environment (Python 3.11)
conda create -n sdft python=3.11 -y
conda activate sdft

# Install the full repository's declared Python dependencies
python -m pip install -e .
```

## :rocket: Quickstart

In the repository root, first prepare the Slurm environment if required by your cluster:

```bash
# Prepare Slurm environment (optional, depending on cluster configuration)
source ./prepare_slurm_env.sh

chmod +x run_sdft.sh

export HF_TOKEN="hf_your_actual_token"
```

The environment helper is cluster-oriented and contains cluster-specific default Conda and scratch paths. **Override `CONDA_ROOT`, `CONDA_ENV`, and `SCRATCH_ROOT` for your cluster**, or skip the helper if the environment is already configured. Run `run_sdft.sh` from the project root, where `configs/deepspeed/zero3.json` and the Python entry points are available.

### Training configurations

`MODEL_SIZE=4b` selects **full fine-tuning**; `MODEL_SIZE=9b` selects **LoRA**. The wrapper launches on four GPUs with ZeRO-3.

<details>
<summary><strong>Qwen3.5-4B — full fine-tuning, EMA teacher</strong></summary>

```bash
MODEL_SIZE=4b TEACHER_KIND=ema LR=1e-5 ./run_sdft.sh
```

</details>

<details>
<summary><strong>Qwen3.5-4B — full fine-tuning, base teacher</strong></summary>

```bash
MODEL_SIZE=4b TEACHER_KIND=base LR=1e-5 ./run_sdft.sh
```

</details>

<details>
<summary><strong>Qwen3.5-9B — LoRA, EMA teacher (r=64, α=128)</strong></summary>

```bash
MODEL_SIZE=9b TEACHER_KIND=ema LR=1e-4 \
LORA_R=64 LORA_ALPHA=128 ./run_sdft.sh
```

</details>

<details>
<summary><strong>Qwen3.5-9B — LoRA, base teacher (r=64, α=128)</strong></summary>

```bash
MODEL_SIZE=9b TEACHER_KIND=base LR=1e-4 \
LORA_R=64 LORA_ALPHA=128 ./run_sdft.sh
```

</details>

### Evaluation and resumption

<details>
<summary><strong>Evaluate an existing checkpoint without training</strong></summary>

Replace `CKPT` with the exact checkpoint directory:

```bash
# 4B full — EMA / base
MODE=eval MODEL_SIZE=4b TEACHER_KIND=ema CKPT=/path/to/checkpoint ./run_sdft.sh
MODE=eval MODEL_SIZE=4b TEACHER_KIND=base CKPT=/path/to/checkpoint ./run_sdft.sh

# 9B LoRA — EMA / base
MODE=eval MODEL_SIZE=9b TEACHER_KIND=ema CKPT=/path/to/checkpoint ./run_sdft.sh
MODE=eval MODEL_SIZE=9b TEACHER_KIND=base CKPT=/path/to/checkpoint ./run_sdft.sh
```

When evaluating a LoRA checkpoint, keep the LoRA configuration (`LORA_R`, `LORA_ALPHA`, etc.) consistent with the original run.

</details>

<details>
<summary><strong>Resume training from a checkpoint</strong></summary>

Replace `RESUME` with the checkpoint directory:

```bash
MODEL_SIZE=4b TEACHER_KIND=ema RESUME=/path/to/checkpoint ./run_sdft.sh
MODEL_SIZE=4b TEACHER_KIND=base RESUME=/path/to/checkpoint ./run_sdft.sh

MODEL_SIZE=9b TEACHER_KIND=ema RESUME=/path/to/checkpoint ./run_sdft.sh
MODEL_SIZE=9b TEACHER_KIND=base RESUME=/path/to/checkpoint ./run_sdft.sh
```

For 9B LoRA, also pass the same LoRA settings used in the original run. The EMA teacher requires its saved state for an exact resume.

</details>

<details>
<summary><strong>Advanced launcher settings and overrides</strong></summary>

The wrapper forwards the following environment settings to the training/evaluation entry point.

| Variable | Default | Purpose |
|:--|:--|:--|
| `MODE` | `train` | `train` or `eval` |
| `MODEL_SIZE` | `4b` | `4b` full FT or `9b` LoRA |
| `TEACHER_KIND` | `ema` | `ema` or `base` in this launcher |
| `BS` / `GA` | `8` / `1` | Per-device batch size / gradient accumulation |
| `NUM_TRAIN_EPOCHS` | `1` | Number of epochs |
| `LR` | `1e-5` (4B), `1e-4` (9B) | Learning rate |
| `SAVE_STEPS` | `100` | Checkpoint cadence in training mode |
| `TEACHER_UPDATE_RATE` | `0.01` | EMA update rate, when using EMA |
| `TEACHER_SYNC_STEPS` | `5` | EMA synchronization interval |
| `DISTILLATION_MODE` | `full_logits` | Distillation-logit selection |
| `DISTILLATION_ALPHA` | `0.0` | Forward KL at `0.0` with full logits |
| `LORA_R` / `LORA_ALPHA` | `64` / `128` | 9B LoRA adapter hyperparameters |
| `LORA_DROPOUT` | `0.05` | 9B LoRA dropout |
| `LORA_TARGETS` | `all-linear` | LoRA target modules |
| `EVAL_BS` | `8` (4B), `4` (9B) | Evaluation batch size |
| `EVAL_MAX_NEW_TOKENS` | `1024` | Generation cap in eval-only mode |
| `DEMONSTRATION_TEMPLATE` | unset in wrapper | Optional override forwarded to the training script |
| `TRAIN_OUT` | `SDFT_OUTPUT_ROOT/<run-name>` | Training output directory |
| `EVAL_RESULTS_DIR` | `checkpoint_evals/<run-name>` | New evaluation artifacts |
| `MASTER_PORT` | `29500` | Distributed launch port |

The launcher uses `max_prompt_length=1024`, `max_completion_length=512`, cosine LR scheduling, bf16, and gradient checkpointing.

Example custom run:

```bash
MODEL_SIZE=9b TEACHER_KIND=ema LR=5e-5 \
LORA_R=64 LORA_ALPHA=128 \
TEACHER_UPDATE_RATE=0.01 TEACHER_SYNC_STEPS=1 \
BS=8 GA=1 SAVE_STEPS=100 ./run_sdft.sh
```

</details>

## :bar_chart: ToolAlpaca evaluation results

Experiments below used **4 NVIDIA L40S GPUs (48 GB each)** and **DeepSpeed ZeRO-3** with bf16. A working DeepSpeed ZeRO-3 configuration for NVIDIA L40S GPUs (48 GB) is provided in configs/deepspeed/zero3.json. Both model families were evaluated on **68 ToolAlpaca test examples**, with greedy generation (`temperature=0`, `max_new_tokens=1024`). Each cell links to the recorded metric JSON and complete generated responses.

> [!IMPORTANT]
> These are archived single-run checkpoint results, not multi-seed averages or results guaranteed by the current default configuration.
**Learning rate:** `1e-5` · **Method:** full fine-tuning · **Teacher comparison:** base vs EMA. The best archived EMA checkpoint is **48/68 (70.6%)** at step 1011; the best base checkpoints are **38/68 (55.9%)** at steps 300 and 900.

| Checkpoint | Base teacher | EMA teacher |
|:--|--:|--:|
| Before training | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/eval_before_results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/eval_before_responses.json)) | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/eval_before_results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/eval_before_responses.json)) |
| 100 | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-100-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-100-responses.json)) | 54.4% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-100-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-100-responses.json)) |
| 200 | 50.0% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-200-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-200-responses.json)) | 58.8% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-200-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-200-responses.json)) |
| 300 | **55.9%** ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-300-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-300-responses.json)) | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-300-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-300-responses.json)) |
| 400 | 52.9% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-400-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-400-responses.json)) | 63.2% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-400-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-400-responses.json)) |
| 500 | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-500-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-500-responses.json)) | 57.4% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-500-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-500-responses.json)) |
| 600 | 52.9% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-600-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-600-responses.json)) | 64.7% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-600-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-600-responses.json)) |
| 700 | 52.9% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-700-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-700-responses.json)) | 63.2% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-700-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-700-responses.json)) |
| 800 | 45.6% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-800-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-800-responses.json)) | 64.7% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-800-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-800-responses.json)) |
| 900 | **55.9%** ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-900-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-900-responses.json)) | 67.6% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-900-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-900-responses.json)) |
| 1000 | 48.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1000-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1000-responses.json)) | 64.7% ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1000-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1000-responses.json)) |
| 1011 | 48.5% ([results](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1011-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-base-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1011-responses.json)) | **70.6%** ([results](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1011-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-4b-ema-zero3-4gpu-1--LR-1e-5-epoch/checkpoint-1011-responses.json)) |

### Qwen3.5-9B — LoRA

**Learning rate:** `1e-4` · **LoRA:** rank `64`, alpha `128` · **Teacher comparison:** base vs EMA. The best archived EMA checkpoints are **46/68 (67.6%)** at steps 900 and 1000; the best base checkpoint is **45/68 (66.2%)** at step 1000.

| Checkpoint | Base teacher | EMA teacher |
|:--|--:|--:|
| Before training | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/eval_before_results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/eval_before_responses.json)) | 51.5% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/eval_before_results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/eval_before_responses.json)) |
| 100 | 54.4% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-100-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-100-responses.json)) | 54.4% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-100-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-100-responses.json)) |
| 200 | 60.3% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-200-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-200-responses.json)) | 57.4% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-200-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-200-responses.json)) |
| 300 | 64.7% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-300-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-300-responses.json)) | 64.7% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-300-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-300-responses.json)) |
| 400 | 63.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-400-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-400-responses.json)) | 66.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-400-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-400-responses.json)) |
| 500 | 57.4% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-500-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-500-responses.json)) | 66.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-500-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-500-responses.json)) |
| 600 | 57.4% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-600-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-600-responses.json)) | 66.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-600-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-600-responses.json)) |
| 700 | 55.9% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-700-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-700-responses.json)) | 63.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-700-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-700-responses.json)) |
| 800 | 63.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-800-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-800-responses.json)) | 66.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-800-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-800-responses.json)) |
| 900 | 61.8% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-900-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-900-responses.json)) | **67.6%** ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-900-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-900-responses.json)) |
| 1000 | **66.2%** ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1000-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1000-responses.json)) | **67.6%** ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1000-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1000-responses.json)) |
| 1011 | 64.7% ([results](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1011-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-base-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1011-responses.json)) | 66.2% ([results](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1011-results.json) · [responses](chkpt_eval_results/sdft-qwen3.5-9b-ema-lora-r64-zero3-4gpu-lr-1e-4/checkpoint-1011-responses.json)) |

### Inspecting model responses

Individual predictions, gold tool calls, and checkpoint-level metrics are under **[`chkpt_eval_results/`](chkpt_eval_results/)**. The two tables above link every checkpoint directly to its corresponding `*-results.json` and `*-responses.json` files. The archived results directory is separate from the default output location for **new** evaluations (`checkpoint_evals/`).

## :bulb: Implementation notes

- **TRL remains the owner of the SDFT algorithm.** The project-specific extension targets ZeRO-3-safe LoRA EMA handling, exact teacher-state persistence, and checkpoint/evaluation tooling.
- **Forward KL is explicitly configured.** Setting `full_logits` and `DISTILLATION_ALPHA=0.0` requests the full-vocabulary `KL(teacher || student)` objective rather than silently inheriting a different TRL default.
- **Checkpoint compatibility matters.** Resume with the same model, LoRA settings, distributed configuration, and saved teacher state.
- **Generation quality and exact tool-call accuracy differ.** A model may predict valid actions while generating unwanted extra text. Inspect `*-responses.json` alongside accuracy numbers, especially for EMA runs.

## :books: References

- [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)
- [Hugging Face TRL’s SDFT](https://github.com/huggingface/trl/tree/main/trl/experimental/sdft)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
