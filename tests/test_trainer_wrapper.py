import ast
import os

from conftest import REPO_ROOT


def test_trainer_is_thin_upstream_subclass_with_only_required_compat_overrides():
    path = os.path.join(REPO_ROOT, "sdft", "trainer.py")
    src = open(path).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SDFTTrainer")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}

    assert "class SDFTTrainer(TRLSDFTTrainer)" in src
    assert methods == {"__init__", "_setup_teacher_model", "_generate"}
    assert "EMACallback(" in src
    assert "self._use_peft_ema_teacher_adapter()" in src
    assert "super()._setup_teacher_model()" in src
    assert "super()._generate(prompt_ids)" in src
    assert 'self.generation_kwargs["eos_token_id"] = self.eos_token_ids' in src
    assert "self.generation_config.eos_token_id = self.eos_token_ids" in src
    assert "config_eos = getattr(text_config, \"eos_token_id\", None)" in src
    assert "self._tokenizer.eos_token_id" in src
    # Still intentionally tiny: no vendored SDFT loss/generation implementation.
    assert len(src.splitlines()) < 110


def test_repo_targets_existing_trl_1142_environment():
    pyproject = open(os.path.join(REPO_ROOT, "pyproject.toml")).read()
    assert '"trl==1.14.2"' in pyproject
    assert '"trl==1.15.0"' not in pyproject
