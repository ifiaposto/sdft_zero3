"""Static checks for the intentionally small resume path."""

import os

from conftest import REPO_ROOT


def test_train_uses_transformers_get_last_checkpoint_directly():
    src = open(os.path.join(REPO_ROOT, "cli", "tooluse_sdft_train.py")).read()
    assert "from transformers.trainer_utils import get_last_checkpoint" in src
    assert "resume_checkpoint = training_args.resume_from_checkpoint" in src
    assert "get_last_checkpoint(training_args.output_dir)" in src
    assert "trainer.train(resume_from_checkpoint=resume_checkpoint)" in src


def test_resume_module_and_resume_eval_callback_are_removed():
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "checkpoint", "resume.py"))
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "callbacks.py"))
    eval_callback = open(os.path.join(REPO_ROOT, "eval", "callback.py")).read()
    assert "ResumeToolEvalCallback" not in eval_callback
