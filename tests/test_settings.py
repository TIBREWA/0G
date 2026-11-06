import os

from isaac.brain.settings import load_env


def test_fills_unset_values_and_skips_comments(tmp_path, monkeypatch):
    monkeypatch.delenv("ZG_TEST_A", raising=False)
    monkeypatch.delenv("ZG_TEST_B", raising=False)
    monkeypatch.delenv("ZG_TEST_C", raising=False)
    env = tmp_path / ".env"
    env.write_text('# note\nZG_TEST_A=one=two\nexport ZG_TEST_B="quoted"\nnot a setting\nZG_TEST_C=\n')
    load_env(env)
    assert os.environ["ZG_TEST_A"] == "one=two"
    assert os.environ["ZG_TEST_B"] == "quoted"
    assert os.environ["ZG_TEST_C"] == ""


def test_shell_value_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ZG_TEST_A", "shell")
    env = tmp_path / ".env"
    env.write_text("ZG_TEST_A=file\n")
    load_env(env)
    assert os.environ["ZG_TEST_A"] == "shell"


def test_missing_file_is_fine(tmp_path):
    load_env(tmp_path / "absent")
