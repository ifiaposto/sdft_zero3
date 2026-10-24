"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

DeepSpeed-safe LoRA EMA teacher adapter creation, updates, and state restoration.

Creates the frozen teacher adapter before DeepSpeed initialization so ZeRO-3 can
partition its parameters and register the required hooks.

Maintains EMA shadow weights in float32 to avoid losing small updates to bf16 rounding.
Under ZeRO-3, gathers student and teacher adapter parameters for updates on rank 0;
otherwise, each rank updates its own replica.

Exposes EMA state export and restoration for checkpoint persistence, including the
float32 shadow weights and update settings. Supports ZeRO-1/2, DDP, and single-GPU use.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

STATE_FORMAT = "peft_ema_shadow_fp32_v2"


def _is_zero3_param(p: torch.Tensor) -> bool:
    return hasattr(p, "ds_id")


def _full_shape(p: torch.Tensor) -> torch.Size:
    return p.ds_shape if _is_zero3_param(p) else p.shape


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class ZeroSafePEFTAdapterEMA:
    def __init__(
        self,
        model,
        accelerator=None,
        teacher_adapter_name: str = "teacher",
        update_rate: float = 0.01,
        sync_steps: int = 1,
    ):
        self.model = model
        self.accelerator = accelerator
        self.teacher_adapter_name = teacher_adapter_name
        self.update_rate = float(update_rate)
        self.sync_steps = int(sync_steps)

        self.student_adapter_name: str | None = None
        self.teacher_adapter_config = None
        self.zero3: bool | None = None
        self._pairs: list[tuple[str, torch.nn.Parameter, torch.nn.Parameter]] | None = None
        # name of teacher param -> fp32 full tensor. Rank 0 only under ZeRO-3, every rank otherwise.
        self.shadow_weights: dict[str, torch.Tensor] | None = None
        self._initialized = False

        self._add_teacher_adapter()

    # ------------------------------------------------------------------ helpers
    def _peft_model(self):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def _adapter_index(self, name: str, adapter: str) -> int | None:
        parts = name.split(".")
        for i in range(1, len(parts)):
            if parts[i] == adapter and parts[i - 1].startswith("lora_"):
                return i
        return None

    def _swap_adapter(self, name: str, src: str, dst: str) -> str:
        parts = name.split(".")
        i = self._adapter_index(name, src)
        if i is None:
            raise KeyError(f"{name} is not a parameter of adapter {src!r}")
        parts[i] = dst
        return ".".join(parts)

    def _is_rank0(self) -> bool:
        return _global_rank() == 0

    # --------------------------------------------------- adapter creation (pre-DS)
    def _add_teacher_adapter(self) -> None:
        model = self._peft_model()
        active = model.active_adapter
        if isinstance(active, (list, tuple)):
            if len(active) != 1:
                raise ValueError(f"Expected exactly one active student adapter, got {active}")
            active = active[0]
        self.student_adapter_name = active or "default"
        if self.teacher_adapter_name in model.peft_config:
            raise ValueError(f"Adapter {self.teacher_adapter_name!r} already exists on the model.")

        trainable_before = {n for n, p in model.named_parameters() if p.requires_grad}

        self.teacher_adapter_config = copy.deepcopy(model.peft_config[self.student_adapter_name])
        model.add_adapter(self.teacher_adapter_name, self.teacher_adapter_config)
        model.set_adapter(self.student_adapter_name)  # restores requires_grad of the student adapter

        n_teacher = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if self._adapter_index(name, self.teacher_adapter_name) is None:
                    continue
                if _is_zero3_param(p):
                    raise RuntimeError(
                        "The EMA teacher adapter must be created before deepspeed.initialize(); "
                        f"{name} is already a ZeRO-3 parameter."
                    )
                p.requires_grad_(False)
                p.zero_()
                n_teacher += 1
        if n_teacher == 0:
            raise RuntimeError("Adding the teacher adapter created no parameters.")

        if self._is_rank0():
            print(
                f"[ema-lora] ZeroSafePEFTAdapterEMA: created teacher adapter "
                f"'{self.teacher_adapter_name}' ({n_teacher} tensors) before DeepSpeed init, fp32 EMA, "
                f"rate={self.update_rate}, every {self.sync_steps} step(s)",
                flush=True,
            )

        trainable_after = {n for n, p in model.named_parameters() if p.requires_grad}
        if trainable_after != trainable_before:
            added = sorted(trainable_after - trainable_before)[:5]
            removed = sorted(trainable_before - trainable_after)[:5]
            raise RuntimeError(
                f"Adding the teacher adapter changed the trainable parameter set (added={added}, removed={removed})."
            )

    # ------------------------------------------------------- lazy init (post-DS)
    def _build_pairs(self) -> None:
        model = self._peft_model()
        params = dict(model.named_parameters())
        pairs = []
        student_lora = set()
        for name, p in params.items():
            if self._adapter_index(name, self.student_adapter_name) is not None:
                student_lora.add(name)
            if self._adapter_index(name, self.teacher_adapter_name) is None:
                continue
            s_name = self._swap_adapter(name, self.teacher_adapter_name, self.student_adapter_name)
            s = params.get(s_name)
            if s is None:
                raise RuntimeError(f"No student parameter {s_name} for teacher parameter {name}")
            if tuple(_full_shape(s)) != tuple(_full_shape(p)):
                raise RuntimeError(f"Shape mismatch {s_name} {tuple(_full_shape(s))} vs {name} {tuple(_full_shape(p))}")
            pairs.append((name, s, p))
        paired_students = {
            self._swap_adapter(n, self.teacher_adapter_name, self.student_adapter_name) for n, _, _ in pairs
        }
        if paired_students != student_lora:
            missing = sorted(student_lora - paired_students)[:5]
            raise RuntimeError(f"Student adapter params without a teacher counterpart: {missing}")
        self._pairs = pairs
        self.zero3 = any(_is_zero3_param(t) for _, _, t in pairs)

    def _device(self) -> torch.device:
        if self.accelerator is not None:
            return self.accelerator.device
        return self._pairs[0][2].device

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._build_pairs()
        if (not self.zero3) or self._is_rank0():
            dev = self._device()
            # Teacher adapter is zero-initialised, so the shadow starts at zero too (same as TRL).
            self.shadow_weights = {
                name: torch.zeros(tuple(_full_shape(t)), dtype=torch.float32, device=dev) for name, _, t in self._pairs
            }
        self._initialized = True

    # ---------------------------------------------------------------- EMA step
    @torch.no_grad()
    def _update_and_write(self) -> None:
        a = self.update_rate
        for name, s, t in self._pairs:
            sh = self.shadow_weights[name]
            sh.mul_(1.0 - a).add_(s.data.to(device=sh.device, dtype=torch.float32), alpha=a)
            t.data.copy_(sh.to(device=t.device, dtype=t.dtype))

    @torch.no_grad()
    def _write_shadow_only(self) -> None:
        for name, _, t in self._pairs:
            t.data.copy_(self.shadow_weights[name].to(device=t.device, dtype=t.dtype))

    def on_train_begin(self, args, state, control, **kwargs):
        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        self._ensure_initialized()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.sync_steps != 0:
            return control
        self._ensure_initialized()
        if self.zero3:
            import deepspeed

            params = [s for _, s, _ in self._pairs] + [t for _, _, t in self._pairs]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if self._is_rank0():
                    self._update_and_write()
        else:
            self._update_and_write()
        return control

    # ------------------------------------------------------- checkpoint support
    def ema_state_dict(self) -> dict[str, Any] | None:
        """Full fp32 EMA state. Returned on global rank 0 (None elsewhere under ZeRO-3). No collectives."""
        self._ensure_initialized()
        if self.shadow_weights is None:
            return None
        return {
            "format": STATE_FORMAT,
            "teacher_adapter_name": self.teacher_adapter_name,
            "student_adapter_name": self.student_adapter_name,
            "update_rate": self.update_rate,
            "sync_steps": self.sync_steps,
            "shadow_weights": {k: v.detach().cpu() for k, v in self.shadow_weights.items()},
        }

    @torch.no_grad()
    def load_ema_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore EMA shadow + teacher adapter. Collective under ZeRO-3: call on EVERY rank."""
        self._ensure_initialized()
        fmt = payload.get("format")
        if fmt != STATE_FORMAT:
            raise RuntimeError(
                f"EMA teacher state has format {fmt!r}, expected {STATE_FORMAT!r}. It was written by a different "
                "EMA implementation (e.g. TRL's PEFTAdapterEMACallback) and cannot be restored exactly."
            )
        loaded = payload["shadow_weights"]
        expected = {name for name, _, _ in self._pairs}
        missing = sorted(expected - set(loaded))
        extra = sorted(set(loaded) - expected)
        if missing or extra:
            raise RuntimeError(f"PEFT EMA key mismatch. missing={missing[:5]} extra={extra[:5]}")
        for name, _, t in self._pairs:
            if tuple(loaded[name].shape) != tuple(_full_shape(t)):
                raise RuntimeError(f"PEFT EMA shape mismatch for {name}")

        if self.shadow_weights is not None:
            for name, sh in self.shadow_weights.items():
                sh.copy_(loaded[name].to(device=sh.device, dtype=torch.float32))

        if self.zero3:
            import deepspeed

            with deepspeed.zero.GatheredParameters([t for _, _, t in self._pairs], modifier_rank=0):
                if self._is_rank0():
                    self._write_shadow_only()
        else:
            self._write_shadow_only()
