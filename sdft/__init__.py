"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Public API for the core SDFT package.
"""

from importlib import import_module

__all__ = [
    "SDFTTrainer",
    "canonicalize_privileged_context",
    "can_prepare_privileged_context",
    "prepare_split",
]

_EXPORTS = {
    "SDFTTrainer": (".trainer", "SDFTTrainer"),
    "canonicalize_privileged_context": (".data", "canonicalize_privileged_context"),
    "can_prepare_privileged_context": (".data", "can_prepare_privileged_context"),
    "prepare_split": (".data", "prepare_split"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
