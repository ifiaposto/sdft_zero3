"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================

Shared runtime and filesystem utilities.

Exports the DeepSpeed checkpoint-loading patch for memory-mapped checkpoint restoration.
"""

from .deepspeed_mmap import patch_deepspeed_checkpoint_mmap

__all__ = ["patch_deepspeed_checkpoint_mmap"]
