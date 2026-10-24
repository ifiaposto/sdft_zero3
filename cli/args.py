"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Script-level SDFT and data arguments, parsed alongside SDFTConfig and ModelConfig by TrlParser.

Adapted from TRL's SDFTScriptArguments and extended with a flag to disable automatic
checkpoint resumption. The demonstration template default is supplied by the task-specific CLI.

Original implementation:

https://github.com/huggingface/trl/blob/main/examples/sdft_privileged_context/sdft_privileged_context.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trl import ScriptArguments


@dataclass
class SDFTScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides `dataset_name`."},
    )
    privileged_context_column: str = field(
        default="privileged_context",
        metadata={"help": "Column containing precomputed privileged context for SDFT."},
    )
    golden_response_column: str = field(
        default="golden_response",
        metadata={"help": "Column containing demonstration responses used to build privileged context."},
    )
    demonstration_template: str | None = field(
        default=None,
        metadata={"help": "Optional template used to build privileged context from demonstration content. Task-specific CLIs may provide a default."},
    )
    no_auto_resume: bool = field(
        default=False,
        metadata={
            "help": "Start from step 0 even if `output_dir` already contains checkpoint-* directories. "
            "By default the newest checkpoint is resumed automatically so a preempted job can be "
            "requeued unchanged. An explicit --resume_from_checkpoint always wins over both."
        },
    )
