import ast
import os

from conftest import REPO_ROOT


def test_custom_evaluator_uses_multi_eos_helper():
    src = open(os.path.join(REPO_ROOT, "eval", "toolalpaca.py")).read()
    assert "def _eos_token_ids(" in src
    assert "eos_token_id=_eos_token_ids(trainer, tokenizer)" in src
    assert "eos_token_id=tokenizer.eos_token_id" not in src


def test_custom_evaluator_truncates_prompts_on_the_left_like_training():
    src = open(os.path.join(REPO_ROOT, "eval", "toolalpaca.py")).read()
    assert 'tokenizer.truncation_side = "left"' in src
    assert "tokenizer.truncation_side = old_truncation_side" in src
    assert 'padding_side="left"' in src
