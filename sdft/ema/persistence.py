"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

EMA teacher checkpoint saving and restoration under checkpoint_dir/ema_teacher.

Supports:
- Pure-LoRA EMA: saves the float32 shadow weights and update settings in one file.
- Full-model EMA under DeepSpeed ZeRO-3: saves each rank's local parameter shard
  and shared metadata, avoiding a full-model gather.

Uses atomic file writes and validates required files, parameter names, and shapes
during restoration. Full-model ZeRO-3 restoration requires the same number of ranks.

Full-model EMA checkpointing without ZeRO-3 is unsupported.
Call save_ema_teacher and restore_ema_teacher on every rank.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from utils.io import atomic_torch_save, atomic_write_json

EMA_DIRNAME = "ema_teacher"
PEFT_SHADOW_FILE = "peft_ema_shadow.pt"
ZERO3_FORMAT = "zero3_local_param_shards_v1"


def _rank_file(rank: int) -> str:
    return f"rank-{rank:05d}.pt"


def _find_peft_ema_callback(trainer):
    for callback in trainer.callback_handler.callbacks:
        if callback.__class__.__name__ == "EMACallback" and hasattr(callback, "ema_state_dict"):
            return callback
    return None


def _zero3_teacher_params(trainer) -> list[tuple[str, torch.nn.Parameter]]:
    teacher_module = getattr(trainer.teacher_model, "module", None)
    if teacher_module is None:
        return []
    return [(name, p) for name, p in teacher_module.named_parameters() if getattr(p, "ds_tensor", None) is not None]


def _save_peft_shadow(trainer, ema_dir: str, step: int) -> dict[str, Any]:
    ema_cb = _find_peft_ema_callback(trainer)
    if ema_cb is None or not hasattr(ema_cb, "ema_state_dict"):
        raise RuntimeError("EMACallback with PEFT EMA state is unavailable at checkpoint save.")
    payload = ema_cb.ema_state_dict()
    if trainer.accelerator.is_main_process:
        if payload is None:
            raise RuntimeError("PEFT EMA shadow weights are unavailable on the main process at checkpoint save.")
        payload["global_step"] = step
        atomic_torch_save(payload, os.path.join(ema_dir, PEFT_SHADOW_FILE))
    trainer.accelerator.wait_for_everyone()
    return {"kind": "peft_adapter_ema", "path": f"{EMA_DIRNAME}/{PEFT_SHADOW_FILE}"}


def _save_zero3_shards(trainer, ema_dir: str, step: int, zero3_params) -> dict[str, Any]:
    accelerator = trainer.accelerator
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)

    payload = {
        "format": ZERO3_FORMAT,
        "global_step": int(step),
        "world_size": world_size,
        "rank": rank,
        "parameters": {name: param.ds_tensor.detach().cpu().clone() for name, param in zero3_params},
    }
    atomic_torch_save(payload, os.path.join(ema_dir, _rank_file(rank)))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        atomic_write_json(
            os.path.join(ema_dir, "metadata.json"),
            {
                "format": ZERO3_FORMAT,
                "global_step": int(step),
                "world_size": world_size,
                "teacher_update_rate": float(trainer.args.teacher_update_rate),
                "teacher_sync_steps": int(trainer.args.teacher_sync_steps),
                "rank_files": [_rank_file(r) for r in range(world_size)],
            },
        )
    accelerator.wait_for_everyone()
    return {"kind": ZERO3_FORMAT, "path": EMA_DIRNAME, "world_size": world_size}


def save_ema_teacher(trainer, checkpoint_dir: str, step: int) -> dict[str, Any]:
    """Persist the EMA teacher under ``checkpoint_dir/ema_teacher``."""
    if trainer.args.teacher_model_kind != "ema":
        return {"kind": trainer.args.teacher_model_kind, "saved": False}

    ema_dir = os.path.join(checkpoint_dir, EMA_DIRNAME)
    os.makedirs(ema_dir, exist_ok=True)

    if trainer._use_peft_ema_teacher_adapter():
        return _save_peft_shadow(trainer, ema_dir, step)

    zero3_params = _zero3_teacher_params(trainer)
    if zero3_params:
        return _save_zero3_shards(trainer, ema_dir, step, zero3_params)

    raise RuntimeError(
        "Full-model EMA checkpointing without DeepSpeed ZeRO-3 is not supported by this simplified repository."
    )


def _restore_peft_shadow(trainer, ema_dir: str) -> None:
    path = os.path.join(ema_dir, PEFT_SHADOW_FILE)
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing PEFT EMA shadow state: {path}")
    ema_cb = _find_peft_ema_callback(trainer)
    if ema_cb is None or not hasattr(ema_cb, "load_ema_state_dict"):
        raise RuntimeError("EMACallback with PEFT EMA state was not registered before EMA restore.")
    ema_cb.load_ema_state_dict(torch.load(path, map_location="cpu"))
    trainer.accelerator.wait_for_everyone()


def _restore_zero3_shards(trainer, ema_dir: str, metadata_path: str) -> None:
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("format") != ZERO3_FORMAT:
        raise RuntimeError(f"Unsupported EMA metadata format in {metadata_path}: {metadata.get('format')}")

    expected_world = int(metadata["world_size"])
    actual_world = int(trainer.accelerator.num_processes)
    if actual_world != expected_world:
        raise RuntimeError(
            f"EMA checkpoint was saved with world_size={expected_world}, but resume uses {actual_world}. "
            "Exact ZeRO-3 EMA restore requires the same number of ranks."
        )

    rank = int(trainer.accelerator.process_index)
    rank_path = os.path.join(ema_dir, _rank_file(rank))
    if not os.path.isfile(rank_path):
        raise RuntimeError(f"Missing EMA shard for rank {rank}: {rank_path}")
    payload = torch.load(rank_path, map_location="cpu")
    loaded = payload["parameters"]

    current = dict(_zero3_teacher_params(trainer))
    if set(current) != set(loaded):
        missing = sorted(set(current) - set(loaded))
        extra = sorted(set(loaded) - set(current))
        raise RuntimeError(f"EMA ZeRO-3 key mismatch. missing={missing[:5]} extra={extra[:5]}")

    with torch.no_grad():
        for name, param in current.items():
            target = param.ds_tensor
            source = loaded[name]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"EMA ZeRO-3 shard shape mismatch for {name}: saved={tuple(source.shape)} current={tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    trainer.accelerator.wait_for_everyone()


def restore_ema_teacher(trainer, checkpoint_dir: str) -> None:
    """Restore the exact EMA teacher for an EMA training resume."""
    if trainer.args.teacher_model_kind != "ema":
        return

    ema_dir = os.path.join(checkpoint_dir, EMA_DIRNAME)
    if not os.path.isdir(ema_dir):
        raise RuntimeError(
            f"EMA resume requested from {checkpoint_dir}, but {ema_dir} is missing. "
            "Refusing to resume with a reset EMA history."
        )

    if trainer._use_peft_ema_teacher_adapter():
        _restore_peft_shadow(trainer, ema_dir)
        return

    metadata_path = os.path.join(ema_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise RuntimeError(f"Missing ZeRO-3 EMA metadata: {metadata_path}")
    _restore_zero3_shards(trainer, ema_dir, metadata_path)
