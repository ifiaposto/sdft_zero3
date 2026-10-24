import os

from conftest import REPO_ROOT


def _read(*parts):
    return open(os.path.join(REPO_ROOT, *parts)).read()


def test_callbacks_are_split_by_responsibility():
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "callbacks.py"))

    ema_src = _read("sdft", "ema", "callback.py")
    assert "EMACallback" in ema_src
    assert "save_ema_teacher" in ema_src
    assert "restore_ema_teacher" in ema_src
    assert "eval_on_save" not in ema_src
    assert "run_tooluse_eval" not in ema_src

    eval_src = _read("eval", "callback.py")
    assert "TooluseEvalCallback" in eval_src
    assert "eval_on_save" in eval_src
    assert "run_configured_tooluse_eval" in eval_src
    assert "save_ema_teacher" not in eval_src
    assert "restore_ema_teacher" not in eval_src

    combined = ema_src + eval_src
    assert "verify_core_checkpoint" not in combined
    assert ".postsave_pending" not in combined
    assert ".postsave_complete" not in combined
    assert "ResumeToolEvalCallback" not in combined


def test_train_has_no_forced_final_checkpoint_or_hostmem_sampler():
    src = _read("cli", "tooluse_sdft_train.py")
    assert "ensure_final_checkpoint" not in src
    assert "start_sampler" not in src
    assert "set_phase" not in src
    assert "log_mem" not in src
    assert 'os.path.join(output_dir, "final_model")' in src


def test_ema_persistence_drops_non_deepspeed_full_state_path():
    src = _read("sdft", "ema", "persistence.py")
    assert "full_state_dict_v1" not in src
    assert "ema_teacher.pt" not in src
    assert "Full-model EMA checkpointing without DeepSpeed ZeRO-3 is not supported" in src
