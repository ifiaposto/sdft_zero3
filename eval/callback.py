"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

ToolAlpaca evaluation callback and shared evaluation helpers.

Extends the original example's evaluation flow with two callback modes:
- Evaluation after each completed checkpoint save.
- Eval-only execution after checkpoint restoration, using on_train_begin to evaluate
  the restored model and raise EvalOnlyComplete before the first training step.

"""

from __future__ import annotations

import os

from transformers import TrainerCallback

from .toolalpaca import run_tooluse_eval


def has_golden_answers(dataset) -> bool:
    return dataset is not None and "golden_answer" in dataset.column_names


def run_configured_tooluse_eval(
    trainer,
    raw_eval_dataset,
    eval_args,
    metric_prefix: str,
    output_dir,
    artifact_stem=None,
):
    """Run ToolAlpaca with the evaluation settings from ``EvalArguments``."""
    return run_tooluse_eval(
        trainer,
        raw_eval_dataset,
        max_new_tokens=eval_args.eval_max_new_tokens,
        num_examples=eval_args.eval_num_examples,
        metric_prefix=metric_prefix,
        batch_size=eval_args.eval_batch_size,
        output_dir=output_dir,
        artifact_stem=artifact_stem,
    )


def log_eval_metrics(trainer, metrics: dict[str, float], combined: bool = True) -> None:
    trainer.log(metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics, combined=combined)


class EvalOnlyComplete(Exception):
    """Raised after a restored checkpoint has been evaluated, before any training step."""


class TooluseEvalCallback(TrainerCallback):
    """Run ToolAlpaca either after checkpoint saves or once after checkpoint restore.

    ``eval_only_checkpoint`` selects eval-only mode. Otherwise the callback only
    acts on saves when ``eval_args.eval_on_save`` is enabled.
    """

    def __init__(self, trainer, raw_eval_dataset, eval_args, eval_only_checkpoint: str | None = None):
        self.trainer = trainer
        self.raw_eval_dataset = raw_eval_dataset
        self.eval_args = eval_args
        self.eval_only_checkpoint = eval_only_checkpoint
        self._eval_only_done = False

    def _artifact_dir(self, *, eval_only: bool) -> str:
        if self.eval_args.eval_artifact_dir:
            return self.eval_args.eval_artifact_dir
        if eval_only:
            return self.trainer.args.output_dir
        return os.path.join(self.trainer.args.output_dir, "checkpoint_evals")

    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_only_checkpoint is None or self._eval_only_done:
            return control
        self._eval_only_done = True

        accelerator = self.trainer.accelerator
        if accelerator.is_main_process:
            print(
                f"[eval-only] restored checkpoint: {self.eval_only_checkpoint} "
                f"(global_step={state.global_step})",
                flush=True,
            )
            print("[eval-only] running ToolAlpaca evaluation; no training step will run.", flush=True)

        artifact_dir = self._artifact_dir(eval_only=True)
        stem = f"checkpoint-{state.global_step}"
        metrics = run_configured_tooluse_eval(
            self.trainer,
            self.raw_eval_dataset,
            self.eval_args,
            metric_prefix="eval_checkpoint",
            output_dir=artifact_dir,
            artifact_stem=stem,
        )
        log_eval_metrics(self.trainer, metrics, combined=False)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            acc = float(metrics.get("eval_checkpoint/accuracy", 0.0))
            n_ok = int(metrics.get("eval_checkpoint/num_correct", 0))
            n_all = int(metrics.get("eval_checkpoint/num_total", 0))
            print(f"[eval-only] DONE: accuracy={acc:.4f} ({n_ok}/{n_all})", flush=True)
            print(f"[eval-only] results: {os.path.join(artifact_dir, stem + '-results.json')}", flush=True)
            print(f"[eval-only] responses: {os.path.join(artifact_dir, stem + '-responses.json')}", flush=True)

        raise EvalOnlyComplete()

    def on_save(self, args, state, control, **kwargs):
        if not self.eval_args.eval_on_save:
            return control
        if not has_golden_answers(self.raw_eval_dataset):
            raise RuntimeError(
                "--eval_on_save was requested but no ToolAlpaca golden_answer eval split is available."
            )

        step = int(state.global_step)
        artifact_dir = self._artifact_dir(eval_only=False)
        if self.trainer.accelerator.is_main_process:
            os.makedirs(artifact_dir, exist_ok=True)
            print(f"[checkpoint {step}] running ToolAlpaca evaluation", flush=True)
        metrics = run_configured_tooluse_eval(
            self.trainer,
            self.raw_eval_dataset,
            self.eval_args,
            metric_prefix="eval_checkpoint",
            output_dir=artifact_dir,
            artifact_stem=f"checkpoint-{step}",
        )
        log_eval_metrics(self.trainer, metrics, combined=False)
        return control
