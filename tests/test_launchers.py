"""Static checks that launchers match the simplified Python code."""

import ast
import glob
import os
import re

from conftest import REPO_ROOT

OUR_FLAGS = re.compile(
    r"--(eval_(?:num_examples|max_new_tokens|batch_size|on_save|artifact_dir)|no_auto_resume|dataset_path|demonstration_template|"
    r"privileged_context_column|golden_response_column)\b"
)

AUTON_TRAIN = [
    "run_train_4b_teacher_512.sh",
    "run_train_4b_teacher_512_lora.sh",
    "run_train_9b_teacher_512_lora.sh",
]

BABEL_TRAIN = [
    "run_train_4b_teacher_512_full_babel.sh",
    "run_train_4b_teacher_512_lora_babel.sh",
    "run_train_9b_teacher_512_lora_babel.sh",
]


def _dataclass_fields(path: str, class_name: str) -> set[str]:
    tree = ast.parse(open(os.path.join(REPO_ROOT, path)).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {stmt.target.id for stmt in node.body if isinstance(stmt, ast.AnnAssign)}
    raise AssertionError(f"{class_name} not found in {path}")


def _cli_arg_fields() -> set[str]:
    return _dataclass_fields("cli/args.py", "SDFTScriptArguments") | _dataclass_fields("eval/args.py", "EvalArguments")


def test_launcher_flags_exist():
    fields = _cli_arg_fields()
    for sh in glob.glob(os.path.join(REPO_ROOT, "run_*.sh")):
        for flag in OUR_FLAGS.findall(open(sh).read()):
            assert flag in fields, f"{os.path.basename(sh)} passes --{flag}, which the parsed CLI argument dataclasses lack"


def test_launched_python_files_exist():
    for sh in glob.glob(os.path.join(REPO_ROOT, "*.sh")):
        for target in re.findall(r"^\s+(\w+\.py)\s*\\?$", open(sh).read(), re.M):
            assert os.path.isfile(os.path.join(REPO_ROOT, target)), f"{sh} launches missing {target}"


def test_auton_training_launchers_keep_paper_defaults():
    for name in AUTON_TRAIN:
        src = open(os.path.join(REPO_ROOT, name)).read()
        assert 'DISTILLATION_MODE="${DISTILLATION_MODE:-full_logits}"' in src
        assert 'DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.0}"' in src
        assert '--distillation_mode "${DISTILLATION_MODE}"' in src
        assert '--distillation_alpha "${DISTILLATION_ALPHA}"' in src
        assert 'TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.01}"' in src
        assert 'TEACHER_SYNC_STEPS="${TEACHER_SYNC_STEPS:-1}"' in src


def test_training_launchers_use_cli_tooluse_template_by_default_and_allow_override():
    for name in AUTON_TRAIN + BABEL_TRAIN:
        src = open(os.path.join(REPO_ROOT, name)).read()
        assert "DEFAULT_DEMONSTRATION_TEMPLATE=" not in src
        assert "tooluse_sdft_train_entry.py" in src
        # Launchers with a demonstration override pass it conditionally; otherwise the CLI default is used.
        if "DEMONSTRATION_TEMPLATE" in src:
            assert "--demonstration_template" in src


def test_auton_transfer_is_end_only_and_controlled_by_two_flags():
    for name in AUTON_TRAIN:
        src = open(os.path.join(REPO_ROOT, name)).read()
        assert 'SAVE_NEWEST_CHECKPOINT_TO_HOME="${SAVE_NEWEST_CHECKPOINT_TO_HOME:-1}"' in src
        assert 'SAVE_FINAL_MODEL_TO_HOME="${SAVE_FINAL_MODEL_TO_HOME:-1}"' in src
        assert "background_sync" not in src
        assert "SYNC_INTERVAL" not in src
        assert ".partial" not in src
        assert "verify_checkpoint.sh" not in src


def test_auton_supports_fresh_explicit_resume_and_auto_resume():
    for name in AUTON_TRAIN:
        src = open(os.path.join(REPO_ROOT, name)).read()
        assert "FRESH=" in src and "--no_auto_resume" in src
        assert "RESUME=" in src and "--resume_from_checkpoint" in src
        assert "Auto-resume enabled" in src


def test_babel_launchers_are_still_present_and_keep_scratch_resume_verification():
    for name in BABEL_TRAIN:
        path = os.path.join(REPO_ROOT, name)
        assert os.path.isfile(path)
        src = open(path).read()
        assert "verify_checkpoint.sh" in src
        assert "--resume_from_checkpoint" in src


def test_cli_and_entry_names_are_consistent():
    assert os.path.isfile(os.path.join(REPO_ROOT, "cli", "tooluse_eval.py"))
    assert not os.path.exists(os.path.join(REPO_ROOT, "cli", "eval_checkpoint.py"))
    wrapper = open(os.path.join(REPO_ROOT, "tooluse_eval_entry.py")).read()
    assert "from cli.tooluse_eval import main" in wrapper
    train_wrapper = open(os.path.join(REPO_ROOT, "tooluse_sdft_train_entry.py")).read()
    assert "from cli.tooluse_sdft_train import main" in train_wrapper


def test_removed_teacher_context_suffix_flag_is_not_used():
    for sh in glob.glob(os.path.join(REPO_ROOT, "run_train_*.sh")):
        assert "--teacher_context_instruction" not in open(sh).read()
