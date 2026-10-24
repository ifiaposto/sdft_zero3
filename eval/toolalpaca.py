"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

ToolAlpaca evaluation adapted from TRL's SDFT example.

Main changes:
- Configurable mini-batch greedy generation with left padding to reduce peak GPU memory.
- Custom scoring delegated to scoring.py.
- Multiple EOS token IDs, aligned with training.

Original implementation:

https://github.com/huggingface/trl/blob/main/examples/sdft_privileged_context/sdft_privileged_context.py
"""

from __future__ import annotations

import os
from typing import Any

import torch
from trl.models import unwrap_model_for_generation

from utils.io import atomic_write_json
from .scoring import score_tooluse_response


def _extract_prompt_text(prompt: Any) -> str:
    """Return the last user message, or the prompt itself when it is already text."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        for message in reversed(prompt):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
                return content
    return str(prompt)


def _apply_prompt_template(tokenizer, prompt: Any, chat_template_kwargs: dict[str, Any] | None = None) -> str:
    """Format one ToolAlpaca prompt using the same chat-template kwargs as SDFT generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": _extract_prompt_text(prompt)}],
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )


def _eos_token_ids(trainer, tokenizer) -> list[int]:
    """Return the exact EOS set used by SDFT generation.

    The local TRL-1.14.2 wrapper exposes ``trainer.eos_token_ids`` (model/text-config
    EOS IDs plus tokenizer EOS). The fallback mirrors that rule so standalone evaluation
    cannot silently regress to a single tokenizer EOS.
    """
    ids = getattr(trainer, "eos_token_ids", None)
    if ids is None:
        generation_ids = getattr(trainer.model.generation_config, "eos_token_id", None)
        if generation_ids is None:
            ids = []
        elif isinstance(generation_ids, int):
            ids = [generation_ids]
        else:
            ids = list(generation_ids)
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
            ids.insert(0, tokenizer.eos_token_id)
    return list(dict.fromkeys(int(token_id) for token_id in ids if token_id is not None))


def _generate(trainer, prompt_texts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedy-decode `prompt_texts` in chunks of `batch_size` and return the completions.

    Greedy decoding is independent per sequence, so chunking cannot change any score; it only
    bounds the KV cache. Under ZeRO-3 `unwrap_model_for_generation` all-gathers the full
    model, so the gather wraps the whole loop rather than each chunk.

    This can run from `Trainer.on_save()`, where the model is in train mode; the previous
    mode is restored afterwards so training continues exactly where it left off.
    """
    tokenizer = trainer.processing_class
    responses: list[str] = []
    batch_size = max(1, batch_size)

    was_training = trainer.model_wrapped.training
    trainer.model_wrapped.eval()
    try:
        with (
            unwrap_model_for_generation(
                trainer.model_wrapped,
                trainer.accelerator,
                gather_deepspeed3_params=trainer.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            for start in range(0, len(prompt_texts), batch_size):
                # Truncate on the left, like TRL training (`ids[-max_prompt_length:]`): an over-long
                # prompt loses the start of its tool docs but keeps the question and the assistant
                # marker. The tokenizer default ("right") would cut those off instead.
                old_truncation_side = tokenizer.truncation_side
                try:
                    tokenizer.truncation_side = "left"
                    tokenized = tokenizer(
                        text=prompt_texts[start : start + batch_size],
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        truncation=True,
                        max_length=trainer.max_prompt_length,
                        add_special_tokens=False,
                    )
                finally:
                    tokenizer.truncation_side = old_truncation_side
                tokenized = {key: value.to(trainer.accelerator.device) for key, value in tokenized.items()}

                generated = unwrapped_model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # temperature=0 / greedy, as in eval_tooluse.py
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=_eos_token_ids(trainer, tokenizer),
                )

                prompt_length = tokenized["input_ids"].shape[1]
                responses.extend(tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True))
                del tokenized, generated
    finally:
        if was_training:
            trainer.model_wrapped.train()
    return responses


def run_tooluse_eval(
    trainer,
    eval_dataset,
    max_new_tokens: int,
    num_examples: int | None = None,
    metric_prefix: str = "tool_eval",
    batch_size: int = 8,
    output_dir: str | None = None,
    artifact_stem: str | None = None,
) -> dict[str, float]:
    """Evaluate the current model on ToolAlpaca with the authors' metric.

    Returns `{prefix}/accuracy`, `{prefix}/num_correct`, `{prefix}/num_total`. Rank 0 also
    writes every response and the summary to `output_dir` (default `trainer.args.output_dir`):

    - without `artifact_stem`: `{metric_prefix}_responses.json`, `{metric_prefix}_results.json`
    - with `artifact_stem`:    `{artifact_stem}-responses.json`, `{artifact_stem}-results.json`

    Must be called on every rank (generation is collective under ZeRO-3).
    """
    if num_examples is not None:
        eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))

    chat_template_kwargs = getattr(trainer, "chat_template_kwargs", None)
    prompt_texts = [
        _apply_prompt_template(trainer.processing_class, prompt, chat_template_kwargs)
        for prompt in eval_dataset["prompt"]
    ]
    responses = _generate(trainer, prompt_texts, max_new_tokens, batch_size)

    scores: list[int] = []
    records: list[dict[str, Any]] = []
    for example, response in zip(eval_dataset, responses, strict=True):
        golden_answer = example.get("golden_answer") or []
        score, details = score_tooluse_response(response, golden_answer)
        scores.append(score)
        records.append(
            {
                "prompt": _extract_prompt_text(example["prompt"]),
                "response": response,
                "golden_answer": golden_answer,
                **details,
            }
        )

    num_total = len(scores)
    num_correct = int(sum(scores))
    accuracy = num_correct / max(num_total, 1)
    metrics = {
        f"{metric_prefix}/accuracy": accuracy,
        f"{metric_prefix}/num_correct": float(num_correct),
        f"{metric_prefix}/num_total": float(num_total),
    }

    # Every rank holds identical results; only rank 0 writes, to avoid racing on a shared filesystem.
    if trainer.accelerator.is_main_process:
        artifact_dir = output_dir or trainer.args.output_dir
        os.makedirs(artifact_dir, exist_ok=True)
        file_stem = artifact_stem or metric_prefix
        separator = "-" if artifact_stem is not None else "_"
        atomic_write_json(os.path.join(artifact_dir, f"{file_stem}{separator}responses.json"), records)
        atomic_write_json(
            os.path.join(artifact_dir, f"{file_stem}{separator}results.json"),
            {
                "global_step": int(trainer.state.global_step),
                "metric_prefix": metric_prefix,
                "accuracy": accuracy,
                "num_correct": num_correct,
                "num_total": num_total,
                "per_sample_scores": scores,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        )

    trainer.accelerator.wait_for_everyone()
    return metrics
