"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Eval-only entry point for resumable SDFT/DeepSpeed checkpoints on ToolAlpaca.

Uses trainer.train(resume_from_checkpoint=...) to initialize DeepSpeed and restore
the sharded checkpoint. TooluseEvalCallback evaluates in on_train_begin, then raises
EvalOnlyComplete before the first training step. The exception is caught here to
exit without training or saving a model.
"""

from __future__ import annotations

import os

from transformers import AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config, get_quantization_config
from trl.experimental.sdft import SDFTConfig

from sdft import SDFTTrainer
from .args import SDFTScriptArguments
from .tooluse_data import load_dataset_dict, prepare_tooluse_split
from eval import EvalArguments, EvalOnlyComplete, TooluseEvalCallback, has_golden_answers
from utils import patch_deepspeed_checkpoint_mmap

from .tooluse_utils import force_disable_thinking
from .tooluse_sdft_train import _apply_tooluse_defaults


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


def main() -> None:
    parser = TrlParser((SDFTScriptArguments, EvalArguments, SDFTConfig, ModelConfig))
    script_args, eval_args, training_args, model_args = parser.parse_args_and_config()
    _apply_tooluse_defaults(script_args)
    force_disable_thinking(training_args)

    checkpoint = training_args.resume_from_checkpoint
    if not checkpoint:
        raise ValueError("Eval-only mode requires --resume_from_checkpoint /path/to/checkpoint-N")
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isdir(checkpoint):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    patch_deepspeed_checkpoint_mmap(verbose=(rank == 0))

    if model_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required")
    _configure_model_loading(training_args, model_args)
    dataset = load_dataset_dict(script_args)

    if script_args.dataset_test_split not in dataset:
        raise ValueError(
            f"Eval split {script_args.dataset_test_split!r} not found; available splits: {list(dataset.keys())}"
        )
    raw_eval_dataset = dataset[script_args.dataset_test_split]
    if not has_golden_answers(raw_eval_dataset):
        raise ValueError(
            "The ToolAlpaca eval split must contain a `golden_answer` column; "
            f"columns are: {raw_eval_dataset.column_names}"
        )

    train_dataset = prepare_tooluse_split(dataset[script_args.dataset_train_split], script_args)
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
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.add_callback(
        TooluseEvalCallback(trainer, raw_eval_dataset, eval_args, eval_only_checkpoint=checkpoint)
    )

    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except EvalOnlyComplete:
        trainer.accelerator.wait_for_everyone()
        if trainer.accelerator.is_main_process:
            print("[eval-only] exiting without training or saving a model.", flush=True)
        return

    raise RuntimeError("Eval-only callback did not abort trainer.train() as expected; refusing to continue.")


if __name__ == "__main__":
    main()
