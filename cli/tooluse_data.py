"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Tool-use dataset preprocessing adapted from TRL's SDFT example.

Wraps raw string prompts as user chat messages so TRL applies the model's chat template.
Adds validation for required fields and missing demonstration templates.

Original implementation:

https://github.com/huggingface/trl/blob/main/examples/sdft_privileged_context/sdft_privileged_context.py
"""

from __future__ import annotations

from string import Template
from typing import Any

from sdft.data import canonicalize_privileged_context


def _extract_prompt_text(prompt: Any) -> str:
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


def _stringify_golden_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return "\n".join(_stringify_golden_response(item) for item in response)
    return str(response)


def _as_chat_prompt(prompt: Any) -> Any:
    """Wrap ToolAlpaca's raw text prompt as one user turn for TRL chat templating."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _build_privileged_context(
    example: dict[str, Any],
    privileged_context_column: str,
    golden_response_column: str,
    template: Template | None,
) -> dict[str, Any]:
    """Adapt native SDFT or demonstration-format rows to canonical SDFT input."""
    if privileged_context_column in example and example[privileged_context_column] is not None:
        canonical = canonicalize_privileged_context(example, privileged_context_column)
        canonical["prompt"] = _as_chat_prompt(canonical["prompt"])
        return canonical

    if golden_response_column in example:
        if template is None:
            raise ValueError("`demonstration_template` is required when `golden_response` is used.")
        privileged_context = template.safe_substitute(
            orig_content=_extract_prompt_text(example["prompt"]),
            output_text=_stringify_golden_response(example[golden_response_column]),
        )
        return {
            "prompt": _as_chat_prompt(example["prompt"]),
            "privileged_context": str(privileged_context),
        }

    if "teacher_prompt" in example:
        raise ValueError(
            "Datasets for `trl.experimental.sdft` should provide `privileged_context` or `golden_response`, not "
            "`teacher_prompt`."
        )
    raise ValueError("Dataset must contain either `privileged_context` or `golden_response` alongside `prompt`.")


def load_dataset_dict(script_args):
    """Load local or Hub data and require named train/test splits."""
    from datasets import DatasetDict, load_dataset, load_from_disk

    if script_args.dataset_path is not None:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            streaming=script_args.dataset_streaming,
        )
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a dataset with named train/test splits (a DatasetDict).")
    return dataset


def prepare_tooluse_split(dataset, script_args):
    """Map native or ToolAlpaca demonstration rows to canonical SDFT columns."""
    template = Template(script_args.demonstration_template) if script_args.demonstration_template else None
    return dataset.map(
        lambda example: _build_privileged_context(
            example,
            privileged_context_column=script_args.privileged_context_column,
            golden_response_column=script_args.golden_response_column,
            template=template,
        ),
        remove_columns=dataset.column_names,
    )


def can_prepare_tooluse_context(dataset, script_args) -> bool:
    """Whether the split can be adapted to SDFT by this tool-use CLI."""
    columns = set(dataset.column_names)
    return "prompt" in columns and (
        script_args.privileged_context_column in columns or script_args.golden_response_column in columns
    )
