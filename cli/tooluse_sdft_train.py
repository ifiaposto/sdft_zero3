"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

ToolAlpaca training entry point adapted from TRL's SDFT example.

Main additions:
- EMA teacher checkpoint saving and restoration.
- Memory-mapped loading of DeepSpeed checkpoints.
- Custom ToolAlpaca evaluation callback for evaluation on checkpoint saves.

Original implementation:

https://github.com/huggingface/trl/blob/main/examples/sdft_privileged_context/sdft_privileged_context.py
"""

from __future__ import annotations

import json
import os
from string import Template

from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config, get_quantization_config
from trl.experimental.sdft import SDFTConfig

from sdft import SDFTTrainer
from .args import SDFTScriptArguments
from .tooluse_data import can_prepare_tooluse_context, load_dataset_dict, prepare_tooluse_split
from eval import EvalArguments, TooluseEvalCallback, has_golden_answers, log_eval_metrics, run_configured_tooluse_eval
from .tooluse_utils import force_disable_thinking
from sdft.ema import EMACallback
from utils import patch_deepspeed_checkpoint_mmap


# Adapted from the SDFT paper's teacher prompt to discourage verbatim copying.
# Adds "original question above" to redirect the teacher to the task rather than
# continuing the demonstration. Omits the request for a thinking process because
# this setup disables Qwen's thinking mode.

# https://arxiv.org/abs/2601.19897


TOOLUSE_DEMONSTRATION_TEMPLATE = Template(
    """This is an example for a response to the question:
$output_text
Now answer the original question above with a response of your own:"""
)

def _apply_tooluse_defaults(script_args) -> None:
    if not script_args.demonstration_template:
        script_args.demonstration_template = TOOLUSE_DEMONSTRATION_TEMPLATE.template


def _configure_model_loading(training_args, model_args) -> None:
    model_kwargs = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs


def _build_trainer(script_args, training_args, model_args):
    if model_args.model_name_or_path is None:
        raise ValueError("`model_name_or_path` is required.")
    _configure_model_loading(training_args, model_args)
    dataset = load_dataset_dict(script_args)

    train_dataset = prepare_tooluse_split(dataset[script_args.dataset_train_split], script_args)
    raw_eval_dataset = dataset[script_args.dataset_test_split] if script_args.dataset_test_split in dataset else None
    eval_dataset = None
    if (
        training_args.eval_strategy != "no"
        and raw_eval_dataset is not None
        and can_prepare_tooluse_context(raw_eval_dataset, script_args)
    ):
        eval_dataset = prepare_tooluse_split(raw_eval_dataset, script_args)

    # ToolAlpaca is text-only. Passing AutoTokenizer explicitly avoids Qwen3.5's
    # multimodal AutoProcessor / torchvision path in this TRL version.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        truncation_side="left",
        padding_side="left",
        trust_remote_code=training_args.trust_remote_code,
    )

    trainer = SDFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    return trainer, eval_dataset, raw_eval_dataset


def _load_baseline_accuracy(artifact_dir: str, verbose: bool) -> dict[str, float] | None:
    path = os.path.join(artifact_dir, "eval_before_results.json")
    try:
        with open(path) as f:
            accuracy = json.load(f)["accuracy"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        if verbose:
            print(f"[resume] baseline accuracy unavailable at {path} ({exc}); delta will be omitted.", flush=True)
        return None
    if verbose:
        print(f"[resume] loaded baseline accuracy {accuracy:.4f} from {path}", flush=True)
    return {"eval_before/accuracy": accuracy}


def _baseline_metrics(trainer, raw_eval_dataset, eval_args, resume_checkpoint):
    if resume_checkpoint is not None:
        return _load_baseline_accuracy(
            eval_args.eval_artifact_dir or trainer.args.output_dir,
            verbose=trainer.accelerator.is_main_process,
        )
    if not has_golden_answers(raw_eval_dataset):
        return None
    metrics = run_configured_tooluse_eval(
        trainer, raw_eval_dataset, eval_args, "eval_before", eval_args.eval_artifact_dir
    )
    log_eval_metrics(trainer, metrics)
    return metrics


def _final_metrics(trainer, eval_dataset, raw_eval_dataset, eval_args, pretrain_metrics) -> None:
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if not has_golden_answers(raw_eval_dataset):
        return

    metrics = run_configured_tooluse_eval(
        trainer, raw_eval_dataset, eval_args, "eval_after", eval_args.eval_artifact_dir
    )
    if pretrain_metrics is not None:
        before = pretrain_metrics.get("eval_before/accuracy")
        after = metrics.get("eval_after/accuracy")
        if before is not None and after is not None:
            metrics["eval_delta/accuracy"] = after - before
    log_eval_metrics(trainer, metrics)


def _save_final_model(trainer, output_dir: str) -> None:
    final_dir = os.path.join(output_dir, "final_model")
    if trainer.accelerator.is_main_process:
        print(f"[final save] {final_dir}", flush=True)
    trainer.save_model(final_dir)
    if trainer.args.should_save:
        trainer.processing_class.save_pretrained(final_dir)
        trainer.model.config.save_pretrained(final_dir)
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    parser = TrlParser((SDFTScriptArguments, EvalArguments, SDFTConfig, ModelConfig))
    script_args, eval_args, training_args, model_args = parser.parse_args_and_config()
    _apply_tooluse_defaults(script_args)
    force_disable_thinking(training_args)

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    patch_deepspeed_checkpoint_mmap(verbose=(rank == 0))

    trainer, eval_dataset, raw_eval_dataset = _build_trainer(script_args, training_args, model_args)

    resume_checkpoint = training_args.resume_from_checkpoint
    if resume_checkpoint is not None and not os.path.isdir(resume_checkpoint):
        raise ValueError(f"--resume_from_checkpoint points at a missing directory: {resume_checkpoint}")
    if resume_checkpoint is None and not script_args.no_auto_resume and os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        if resume_checkpoint is not None:
            print(f"[resume] {resume_checkpoint}", flush=True)
        elif script_args.no_auto_resume:
            print("[resume] disabled; starting from step 0", flush=True)
        else:
            print("[resume] no checkpoint found; starting from step 0", flush=True)

    if trainer.args.teacher_model_kind == "ema":
        ema_callback = next(
            (callback for callback in trainer.callback_handler.callbacks if isinstance(callback, EMACallback)),
            None,
        )
        if ema_callback is None:
            ema_callback = EMACallback()
            trainer.add_callback(ema_callback)
        ema_callback.configure_checkpointing(trainer, resume_checkpoint)
    if eval_args.eval_on_save:
        trainer.add_callback(TooluseEvalCallback(trainer, raw_eval_dataset, eval_args))

    pretrain_metrics = _baseline_metrics(trainer, raw_eval_dataset, eval_args, resume_checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    _final_metrics(trainer, eval_dataset, raw_eval_dataset, eval_args, pretrain_metrics)
    _save_final_model(trainer, training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)


if __name__ == "__main__":
    main()
