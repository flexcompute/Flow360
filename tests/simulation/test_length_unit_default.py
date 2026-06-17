"""Regression tests for the deprecation of the default project length unit."""

import flow360.component.utils as utils_mod
from flow360.component.utils import resolve_length_unit


def test_resolve_length_unit_warns_when_unset(monkeypatch):
    messages = []
    monkeypatch.setattr(utils_mod.log, "warning", lambda msg, *a, **k: messages.append(str(msg)))
    assert resolve_length_unit(None) == "m"
    assert any("DeprecationWarning" in m and "length_unit" in m for m in messages)


def test_resolve_length_unit_silent_when_set(monkeypatch):
    messages = []
    monkeypatch.setattr(utils_mod.log, "warning", lambda msg, *a, **k: messages.append(str(msg)))
    assert resolve_length_unit("mm") == "mm"
    assert messages == []
