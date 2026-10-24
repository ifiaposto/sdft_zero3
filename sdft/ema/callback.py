"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Trainer callback for EMA teacher updates and checkpoint persistence.

- Handles DeepSpeed-safe LoRA EMA adapter creation and updates via ZeroSafePEFTAdapterEMA.
- Saves and restores EMA teacher state alongside Trainer checkpoints for both LoRA
  and full-model EMA teachers.
"""

from __future__ import annotations

import os

from transformers import TrainerCallback

from .peft_ema import ZeroSafePEFTAdapterEMA
from .persistence import restore_ema_teacher, save_ema_teacher


class EMACallback(TrainerCallback):
    """EMA update/persistence callback used by this project.

    Pass ``model`` during SDFT trainer setup for the pure-LoRA EMA path so the
    teacher adapter is created before DeepSpeed initialization. For full-model
    EMA, upstream TRL owns the update callback and this local callback is added
    later only for exact checkpoint save/restore.
    """

    def __init__(
        self,
        model=None,
        accelerator=None,
        teacher_adapter_name: str = "teacher",
        update_rate: float = 0.01,
        sync_steps: int = 1,
    ):
        self.trainer = None
        self.resume_checkpoint: str | None = None
        self.peft_ema = None
        if model is not None:
            self.peft_ema = ZeroSafePEFTAdapterEMA(
                model=model,
                accelerator=accelerator,
                teacher_adapter_name=teacher_adapter_name,
                update_rate=update_rate,
                sync_steps=sync_steps,
            )

    def configure_checkpointing(self, trainer, resume_checkpoint: str | None) -> None:
        self.trainer = trainer
        self.resume_checkpoint = resume_checkpoint

    def ema_state_dict(self):
        if self.peft_ema is None:
            return None
        return self.peft_ema.ema_state_dict()

    def load_ema_state_dict(self, payload) -> None:
        if self.peft_ema is None:
            raise RuntimeError("PEFT EMA state was provided to a checkpoint-only EMA callback.")
        self.peft_ema.load_ema_state_dict(payload)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.peft_ema is not None:
            control = self.peft_ema.on_train_begin(args, state, control, **kwargs)

        if self.trainer is not None and self.resume_checkpoint is not None:
            if self.trainer.accelerator.is_main_process:
                print(f"[ema resume] restoring EMA teacher from {self.resume_checkpoint}", flush=True)
            restore_ema_teacher(self.trainer, self.resume_checkpoint)
            if self.trainer.accelerator.is_main_process:
                print("[ema resume] EMA teacher restored.", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.peft_ema is None:
            return control
        return self.peft_ema.on_step_end(args, state, control, **kwargs)

    def on_save(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        step = int(state.global_step)
        checkpoint_dir = os.path.join(self.trainer.args.output_dir, f"checkpoint-{step}")
        if self.trainer.accelerator.is_main_process:
            print(f"[checkpoint {step}] saving EMA teacher", flush=True)
        save_ema_teacher(self.trainer, checkpoint_dir, step)
        return control
