"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Dataset preparation & validation for examples with precomputed privileged context.
"""

from __future__ import annotations

from typing import Any


def canonicalize_privileged_context(
    example: dict[str, Any],
    privileged_context_column: str = "privileged_context",
) -> dict[str, Any]:
    """Return one example in the canonical ``prompt + privileged_context`` SDFT shape."""
    if "prompt" not in example:
        raise ValueError("Dataset must contain a `prompt` column.")
    if privileged_context_column not in example or example[privileged_context_column] is None:
        if "teacher_prompt" in example:
            raise ValueError(
                "Datasets for `trl.experimental.sdft` should provide `privileged_context`, not `teacher_prompt`."
            )
        raise ValueError(
            f"Dataset must contain a non-null `{privileged_context_column}` column alongside `prompt`."
        )

    return {
        "prompt": example["prompt"],
        "privileged_context": str(example[privileged_context_column]),
    }


def prepare_split(dataset, privileged_context_column: str = "privileged_context"):
    """Map a raw split with precomputed privileged context to canonical SDFT columns."""
    return dataset.map(
        lambda example: canonicalize_privileged_context(example, privileged_context_column),
        remove_columns=dataset.column_names,
    )


def can_prepare_privileged_context(dataset, privileged_context_column: str = "privileged_context") -> bool:
    """Whether a split already contains the canonical SDFT input columns."""
    columns = set(dataset.column_names)
    return "prompt" in columns and privileged_context_column in columns
