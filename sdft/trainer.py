"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Extended wrapper around TRL's SDFTTrainer.

Delegates the SDFT algorithm, model loading, loss, and generation to TRL.

Main changes:
- Uses the local ZeRO-3-safe callback for pure-LoRA EMA teachers.
- Accepts EOS IDs from the model generation config, text config, and tokenizer.
- Trims completions after the first valid EOS to exclude subsequent padding from the loss.

EMA teacher checkpoint saving and restoration are handled separately in sdft.ema.

Original implementation:

https://github.com/huggingface/trl/blob/main/trl/experimental/sdft/sdft_trainer.py
"""

from trl.experimental.sdft import SDFTTrainer as TRLSDFTTrainer

from .ema import EMACallback


class SDFTTrainer(TRLSDFTTrainer):
    """TRL 1.14.2 SDFTTrainer with ZeRO-safe LoRA EMA and Qwen multi-EOS support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TRL 1.14.2 generates with tokenizer.eos_token_id only. Qwen3.5 has
        # tokenizer EOS=<|im_end|> while its text config also declares
        # <|endoftext|> as EOS. Keep both as valid stopping IDs.
        eos_ids: list[int] = []

        generation_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(generation_eos, int):
            eos_ids.append(generation_eos)
        elif generation_eos is not None:
            eos_ids.extend(int(token_id) for token_id in generation_eos if token_id is not None)

        text_config = self.model.config.get_text_config()
        config_eos = getattr(text_config, "eos_token_id", None)
        if isinstance(config_eos, int):
            eos_ids.append(config_eos)
        elif config_eos is not None:
            eos_ids.extend(int(token_id) for token_id in config_eos if token_id is not None)

        if self._tokenizer.eos_token_id is not None:
            eos_ids.append(int(self._tokenizer.eos_token_id))

        self.eos_token_ids = list(dict.fromkeys(eos_ids))
        if self.eos_token_ids:
            self.generation_kwargs["eos_token_id"] = self.eos_token_ids
            self.generation_config.eos_token_id = self.eos_token_ids

    def _setup_teacher_model(self) -> None:
        """Swap only TRL's pure-LoRA EMA callback; delegate every other teacher mode upstream."""
        if not self._use_peft_ema_teacher_adapter():
            super()._setup_teacher_model()
            return

        # TRL calls this after HF Trainer initialization but before trainer.train() invokes
        # deepspeed.initialize(). Creating the teacher adapter here lets ZeRO-3 partition it
        # and install hooks on it from the start.
        self.add_callback(
            EMACallback(
                model=self.model,
                teacher_adapter_name="teacher",
                update_rate=self.args.teacher_update_rate,
                sync_steps=self.args.teacher_sync_steps,
                accelerator=self.accelerator,
            )
        )
        self.teacher_model = self.model

    def _generate(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        """Use upstream generation, then trim each completion at the first valid EOS.

        TRL 1.14.2 masks completions using only ``tokenizer.eos_token_id``. When
        generation stops on Qwen's other valid EOS, the returned tensor can contain
        padding after that EOS; trimming here keeps those padded tokens out of the SDFT
        completion mask/loss without copying the upstream generation implementation.
        """
        prompt_ids_out, completion_ids_list = super()._generate(prompt_ids)
        if not self.eos_token_ids:
            return prompt_ids_out, completion_ids_list

        eos_ids = set(self.eos_token_ids)
        trimmed: list[list[int]] = []
        for completion in completion_ids_list:
            end = len(completion)
            for idx, token_id in enumerate(completion):
                if token_id in eos_ids:
                    end = idx + 1
                    break
            trimmed.append(completion[:end])
        return prompt_ids_out, trimmed
