from unittest.mock import patch

import flow360.version_utils as vu


def reset_warning_flag():
    vu._WARNED_PRERELEASE = False  # pylint: disable=protected-access


def test_is_prerelease_version_detects_beta():
    assert vu.is_prerelease_version("1.0.0b1")
    assert not vu.is_prerelease_version("1.0.0")


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
