import sys

from click.testing import CliRunner


def test_import_flow360_does_not_eagerly_import_heavy_dependencies(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    sys.modules.pop("flow360", None)
    sys.modules.pop("flow360._api", None)
    sys.modules.pop("pandas", None)

    import flow360  # pylint: disable=import-outside-toplevel,import-error

    assert "pandas" not in sys.modules
    assert hasattr(flow360, "Env")


def test_import_flow360_cli_app_does_not_eagerly_import_sdk_command_modules(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    for module_name in (
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.project",
        "flow360.cli.folder",
        "flow360.cloud.flow360_requests",
    ):
        sys.modules.pop(module_name, None)

    import flow360.cli.app  # pylint: disable=import-outside-toplevel,import-error,unused-import

    assert "flow360.cli.project" not in sys.modules
    assert "flow360.cli.folder" not in sys.modules
    assert "flow360.cloud.flow360_requests" not in sys.modules


def test_flow360_root_help_does_not_eagerly_import_sdk_command_modules(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    for module_name in (
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.project",
        "flow360.cli.folder",
        "flow360.cloud.flow360_requests",
    ):
        sys.modules.pop(module_name, None)

    from flow360.cli import (
        flow360,  # pylint: disable=import-outside-toplevel,import-error
    )

    result = CliRunner().invoke(flow360, ["--help"])

    assert result.exit_code == 0
    assert "flow360.cli.project" not in sys.modules
    assert "flow360.cli.folder" not in sys.modules
    assert "flow360.cloud.flow360_requests" not in sys.modules
