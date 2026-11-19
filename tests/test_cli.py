import sys

import pytest

from src import cli


def run_with(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])


def test_get_converts_type_and_default(monkeypatch):
    run_with(monkeypatch, "--limit", "300")
    assert cli.get("limit", 0) == 300
    assert cli.get("k", 5) == 5          # not typed -> default
    assert cli.get("repo") is None


def test_flag_and_list(monkeypatch):
    run_with(monkeypatch, "--no-header", "--indexes", "regular", "heading", "--k", "3")
    assert cli.flag("no-header") and not cli.flag("allow-missing")
    assert cli.get_list("indexes", ["x"]) == ["regular", "heading"]
    assert cli.get_list("other", ["x"]) == ["x"]


def test_typo_and_bad_number_stop(monkeypatch):
    run_with(monkeypatch, "--limt", "3")
    with pytest.raises(SystemExit):
        cli.check("limit")
    run_with(monkeypatch, "--limit", "abc")
    with pytest.raises(SystemExit):
        cli.get("limit", 0)
