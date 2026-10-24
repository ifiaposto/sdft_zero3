"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Shared ToolAlpaca/Qwen CLI configuration helpers.

Forces Qwen's thinking mode off during training and evaluation for a  direct-response setup. 
Preserves other chat-template settings.
"""

from __future__ import annotations

import json
from typing import Any


def force_disable_thinking(training_args) -> dict[str, Any]:
    """Force Qwen direct-response formatting while preserving other chat-template kwargs."""
    raw_kwargs = getattr(training_args, "chat_template_kwargs", None)
    if raw_kwargs is None:
        chat_kwargs: dict[str, Any] = {}
    elif isinstance(raw_kwargs, str):
        chat_kwargs = json.loads(raw_kwargs)
    else:
        chat_kwargs = dict(raw_kwargs)
        
    # TODO: Make enable_thinking configurable via a CLI argument and compare
    # thinking enabled vs. disabled for tool-call accuracy and truncation rate.

    chat_kwargs["enable_thinking"] = False
    training_args.chat_template_kwargs = chat_kwargs
    return chat_kwargs
