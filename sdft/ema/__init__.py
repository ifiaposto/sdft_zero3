"""
sdft-zero3: Ready-to-run distributed SDFT

Author: Ifigeneia Apostolopoulou
Copyright (c) 2026 Ifigeneia Apostolopoulou

================================================================================================
EMA mechanics and checkpoint persistence.
"""

from .callback import EMACallback
from .persistence import restore_ema_teacher, save_ema_teacher

__all__ = ["EMACallback", "restore_ema_teacher", "save_ema_teacher"]
