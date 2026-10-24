"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Memory-mapped loading of DeepSpeed checkpoints to reduce CPU RAM pressure during resume.

Uses torch.load(..., mmap=True) to load tensor data on demand instead of eagerly reading
it into RAM, limiting memory spikes when multiple ranks restore large checkpoints.
Falls back to DeepSpeed's original loader if memory mapping fails.
"""

from __future__ import annotations


def patch_deepspeed_checkpoint_mmap(verbose: bool = True) -> bool:
    """Ask DeepSpeed checkpoint loads to use ``torch.load(..., mmap=True)`` when possible."""
    try:
        import torch
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
    except Exception as exc:  # pragma: no cover - depends on installed DeepSpeed
        if verbose:
            print(f"[deepspeed] mmap checkpoint patch not applied ({exc})", flush=True)
        return False

    if getattr(TorchCheckpointEngine, "_sdft_mmap_patched", False):
        return True

    original = TorchCheckpointEngine.load

    def _load(self, path, map_location=None):
        try:
            return torch.load(path, map_location=map_location or "cpu", mmap=True, weights_only=False)
        except Exception:
            return original(self, path, map_location=map_location)

    TorchCheckpointEngine.load = _load
    TorchCheckpointEngine._sdft_mmap_patched = True
    if verbose:
        print("[deepspeed] checkpoint loads use mmap=True when supported", flush=True)
    return True
