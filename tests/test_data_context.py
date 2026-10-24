from string import Template

from cli.tooluse_data import _build_privileged_context
from sdft.data import canonicalize_privileged_context

EXPECTED_TEMPLATE = (
    "Here is an example of a correct response:\n"
    "$output_text"
)


def test_default_demonstration_template_is_exact():
    assert EXPECTED_TEMPLATE == EXPECTED_TEMPLATE
    assert "Example response:" not in EXPECTED_TEMPLATE
    assert "Observation" not in EXPECTED_TEMPLATE


def test_default_template_builds_teacher_context_once():
    row = {"prompt": "p", "golden_response": 'Action: weather\nAction Input: {"city": "Boston"}'}
    out = _build_privileged_context(row, "privileged_context", "golden_response", Template(EXPECTED_TEMPLATE))
    assert out["prompt"] == [{"role": "user", "content": "p"}]
    assert out["privileged_context"] == (
        "Here is an example of a correct response:\n"
        'Action: weather\nAction Input: {"city": "Boston"}'
    )


def test_existing_privileged_context_is_used_as_is():
    row = {"prompt": "p", "privileged_context": "teacher info"}
    out = _build_privileged_context(row, "privileged_context", "golden_response", Template(EXPECTED_TEMPLATE))
    assert out["prompt"] == [{"role": "user", "content": "p"}]
    assert out["privileged_context"] == "teacher info"


def test_existing_chat_prompt_is_preserved():
    prompt = [{"role": "user", "content": "p"}]
    row = {"prompt": prompt, "golden_response": "Action: weather"}
    out = _build_privileged_context(row, "privileged_context", "golden_response", Template(EXPECTED_TEMPLATE))
    assert out["prompt"] == prompt


def test_core_sdft_data_accepts_only_canonical_context():
    row = {"prompt": "p", "privileged_context": "teacher info"}
    assert canonicalize_privileged_context(row) == row


def test_tooluse_adaptation_lives_outside_core_sdft():
    import os
    from conftest import REPO_ROOT

    train_cli = open(os.path.join(REPO_ROOT, "cli", "tooluse_sdft_train.py")).read()
    tooluse_data = open(os.path.join(REPO_ROOT, "cli", "tooluse_data.py")).read()
    core_data = open(os.path.join(REPO_ROOT, "sdft", "data.py")).read()

    assert "TOOLUSE_DEMONSTRATION_TEMPLATE" in train_cli
    assert EXPECTED_TEMPLATE in train_cli
    assert "golden_response" in tooluse_data
    assert "demonstration_template" in tooluse_data
    assert "golden_response" not in core_data
    assert "demonstration_template" not in core_data
