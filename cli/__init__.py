"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================
"""


from importlib import import_module

__all__ = ["SDFTScriptArguments", "tooluse_sdft_train_main", "tooluse_eval_main"]

_EXPORTS = {
    "SDFTScriptArguments": (".args", "SDFTScriptArguments"),
    "tooluse_sdft_train_main": (".tooluse_sdft_train", "main"),
    "tooluse_eval_main": (".tooluse_eval", "main"),
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
