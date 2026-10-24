"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Evaluation CLI arguments shared by the tool-use training and eval-only entry points.

Adds configurable evaluation batch size, evaluation after checkpoint saves,
and a separate directory for evaluation results and generated responses.

Original implementation:

https://github.com/huggingface/trl/blob/main/examples/sdft_privileged_context/sdft_privileged_context.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalArguments:
    eval_num_examples: int | None = field(
        default=None,
        metadata={"help": "Evaluate only the first N examples. Defaults to the full evaluation split."},
    )
    eval_max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of generated completion tokens during evaluation."},
    )
    eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "Prompts per generate() call during evaluation. Greedy decoding makes this a memory/throughput setting only."
        },
    )
    eval_on_save: bool = field(
        default=False,
        metadata={"help": "Run evaluation after every completed Trainer checkpoint save."},
    )
    eval_artifact_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory for evaluation result/response artifacts. Defaults to output_dir "
            "(output_dir/checkpoint_evals for per-checkpoint evaluation)."
        },
    )
