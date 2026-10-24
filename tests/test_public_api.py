import ast
import os

from conftest import REPO_ROOT


def _read(rel):
    return open(os.path.join(REPO_ROOT, "sdft", rel)).read()


def _read_repo(rel):
    return open(os.path.join(REPO_ROOT, rel)).read()


def _all_values(rel, *, top_level=False):
    tree = ast.parse(_read_repo(rel) if top_level else _read(rel))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return {elt.value for elt in node.value.elts}
    return None


def test_root_exports_entrypoint_dependencies():
    assert _all_values("__init__.py") == {
        "SDFTTrainer",
        "canonicalize_privileged_context",
        "can_prepare_privileged_context",
        "prepare_split",
    }


def test_subpackage_inits_are_small_and_explicit():
    assert _all_values("cli/__init__.py", top_level=True) == {"SDFTScriptArguments", "tooluse_sdft_train_main", "tooluse_eval_main"}
    assert _all_values("ema/__init__.py") == {
        "EMACallback",
        "restore_ema_teacher",
        "save_ema_teacher",
    }
    assert _all_values("eval/__init__.py", top_level=True) == {
        "EvalArguments",
        "EvalOnlyComplete",
        "TooluseEvalCallback",
        "has_golden_answers",
        "log_eval_metrics",
        "run_configured_tooluse_eval",
        "run_tooluse_eval",
    }
    assert _all_values("utils/__init__.py", top_level=True) == {"patch_deepspeed_checkpoint_mmap"}


def test_removed_checkpoint_modules_are_gone():
    for rel in (
        "checkpoint/final.py",
        "checkpoint/resume.py",
        "checkpoint/verify.py",
        "checkpoint/__init__.py",
        "utils/hostmem.py",
        "utils/dist.py",
    ):
        assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", rel)), rel
    assert not os.path.exists(os.path.join(REPO_ROOT, "cli", "eval_checkpoint.py"))
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "callbacks.py"))
    for rel in (
        "ema/callback.py",
        "ema/persistence.py",
    ):
        assert os.path.exists(os.path.join(REPO_ROOT, "sdft", rel)), rel
    for rel in ("utils/deepspeed_mmap.py", "utils/io.py", "utils/__init__.py"):
        assert os.path.exists(os.path.join(REPO_ROOT, rel)), rel
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "utils"))
    assert os.path.exists(os.path.join(REPO_ROOT, "eval", "callback.py"))
    assert os.path.exists(os.path.join(REPO_ROOT, "cli", "tooluse_sdft_train.py"))
    assert os.path.exists(os.path.join(REPO_ROOT, "cli", "tooluse_eval.py"))
    assert os.path.exists(os.path.join(REPO_ROOT, "cli", "args.py"))
    assert os.path.exists(os.path.join(REPO_ROOT, "cli", "tooluse_data.py"))
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "args.py"))
    assert os.path.exists(os.path.join(REPO_ROOT, "eval", "__init__.py"))
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "evaluation"))


def test_train_cli_helpers_are_private():
    tree = ast.parse(_read_repo("cli/tooluse_sdft_train.py"))
    funcs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" in funcs
    assert all(name == "main" or name.startswith("_") for name in funcs)


def test_train_passes_text_tokenizer_to_sdft_trainer():
    source = _read_repo("cli/tooluse_sdft_train.py")
    assert "AutoTokenizer.from_pretrained(" in source
    assert 'truncation_side="left"' in source
    assert 'padding_side="left"' in source
    assert "processing_class=tokenizer" in source


def test_eval_passes_text_tokenizer_to_sdft_trainer():
    source = _read_repo("cli/tooluse_eval.py")
    assert "AutoTokenizer.from_pretrained(" in source
    assert "processing_class=tokenizer" in source


def test_train_does_not_import_auto_processor_for_text_only_qwen():
    tree = ast.parse(_read_repo("cli/tooluse_sdft_train.py"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "AutoProcessor" not in imported
    assert "AutoTokenizer" in imported


def test_tooluse_thinking_helper_lives_in_cli_not_core_sdft():
    assert os.path.exists(os.path.join(REPO_ROOT, "cli", "tooluse_utils.py"))
    assert "force_disable_thinking" not in _read("data.py")
    assert "force_disable_thinking" not in _read("__init__.py")
    assert "from .tooluse_utils import force_disable_thinking" in _read_repo("cli/tooluse_sdft_train.py")
    assert "from .tooluse_utils import force_disable_thinking" in _read_repo("cli/tooluse_eval.py")


def test_cli_arguments_and_eval_arguments_live_outside_core_sdft():
    cli_args = _read_repo("cli/args.py")
    eval_args = _read_repo("eval/args.py")
    assert not os.path.exists(os.path.join(REPO_ROOT, "sdft", "args.py"))
    assert "class SDFTScriptArguments" in cli_args
    for name in ("eval_num_examples", "eval_max_new_tokens", "eval_batch_size", "eval_on_save", "eval_artifact_dir"):
        assert name not in cli_args
        assert name in eval_args
    assert "tool_eval_" not in eval_args


def test_tooluse_dataset_adaptation_lives_outside_core_sdft():
    core_data = _read("data.py")
    tooluse_data = _read_repo("cli/tooluse_data.py")
    assert "golden_response" not in core_data
    assert "demonstration_template" not in core_data
    assert "golden_response" in tooluse_data
    assert "demonstration_template" in tooluse_data
    assert "from sdft.data" not in _read_repo("eval/toolalpaca.py")
