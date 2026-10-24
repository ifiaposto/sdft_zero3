import json
import os

from utils.io import atomic_write_json, atomic_write_text


def test_atomic_json_roundtrip_and_no_tmp_left(tmp_path):
    path = str(tmp_path / "x.json")
    atomic_write_json(path, {"a": [1, 2]})
    atomic_write_json(path, {"a": [3]})
    assert json.load(open(path)) == {"a": [3]}
    assert not os.path.exists(path + ".tmp")


def test_atomic_text_roundtrip(tmp_path):
    path = str(tmp_path / "x.txt")
    atomic_write_text(path, "step=3\n")
    assert open(path).read() == "step=3\n"
    assert not os.path.exists(path + ".tmp")
