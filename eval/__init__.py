"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================
ToolAlpaca/tool-use evaluation API.
"""

from importlib import import_module

__all__ = [
    "EvalArguments",
    "EvalOnlyComplete",
    "TooluseEvalCallback",
    "has_golden_answers",
    "log_eval_metrics",
    "run_configured_tooluse_eval",
    "run_tooluse_eval",
]

_EXPORTS = {
    "EvalArguments": (".args", "EvalArguments"),
    "EvalOnlyComplete": (".callback", "EvalOnlyComplete"),
    "TooluseEvalCallback": (".callback", "TooluseEvalCallback"),
    "has_golden_answers": (".callback", "has_golden_answers"),
    "log_eval_metrics": (".callback", "log_eval_metrics"),
    "run_configured_tooluse_eval": (".callback", "run_configured_tooluse_eval"),
    "run_tooluse_eval": (".toolalpaca", "run_tooluse_eval"),
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
