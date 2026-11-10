from unittest.mock import patch

import flow360.version_utils as vu


def reset_warning_flag():
    vu._WARNED_PRERELEASE = False  # pylint: disable=protected-access


def test_is_prerelease_version_detects_beta():
    assert vu.is_prerelease_version("1.0.0b1")
    assert not vu.is_prerelease_version("1.0.0")


def test_is_prerelease_version_detects_unofficial_builds():
    # Internal builds are dev releases; detection must not depend on which hex
    # digits happen to appear in the commit sha (the old substring check fired
    # only when the sha contained a "b").
    assert vu.is_prerelease_version("25.11.2.dev20260819143000+gab12cd34")
    assert vu.is_prerelease_version("25.11.2.dev20260819143000+g0a1c2d3e")
    assert vu.is_prerelease_version("25.12.0b1.dev20260819143000+gee11ff22")
    # Not PEP 440 prereleases, but never published officially: a local version
    # segment cannot come from PyPI, and post releases are never cut.
    assert vu.is_prerelease_version("25.11.2+gab12cd34")
    assert vu.is_prerelease_version("25.11.2.post1")


def test_warn_if_prerelease_version_emits_once(monkeypatch):
    reset_warning_flag()
    monkeypatch.setattr(vu, "__version__", "2.0.0b2", raising=False)
    with patch.object(vu.log, "warning") as mock_warning:
        vu.warn_if_prerelease_version()
        vu.warn_if_prerelease_version()
    assert mock_warning.call_count == 1


def test_warn_if_prerelease_version_skips_release(monkeypatch):
    reset_warning_flag()
    monkeypatch.setattr(vu, "__version__", "2.0.0", raising=False)
    with patch.object(vu.log, "warning") as mock_warning:
        vu.warn_if_prerelease_version()
    mock_warning.assert_not_called()


def test_warn_if_prerelease_version_respects_env(monkeypatch):
    reset_warning_flag()
    monkeypatch.setenv("FLOW360_SUPPRESS_BETA_WARNING", "1")
    monkeypatch.setattr(vu, "__version__", "2.0.0b2", raising=False)
    with patch.object(vu.log, "warning") as mock_warning:
        vu.warn_if_prerelease_version()
    mock_warning.assert_not_called()
