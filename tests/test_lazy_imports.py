import sys

import toml
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
        "flow360.cli.assets",
        "flow360.cli.draft",
        "flow360.cli.folder",
        "flow360.cli.wait",
        "flow360.cloud.flow360_requests",
    ):
        sys.modules.pop(module_name, None)

    import flow360.cli.app  # pylint: disable=import-outside-toplevel,import-error,unused-import

    assert "flow360.cli.project" not in sys.modules
    assert "flow360.cli.assets" not in sys.modules
    assert "flow360.cli.draft" not in sys.modules
    assert "flow360.cli.folder" not in sys.modules
    assert "flow360.cli.wait" not in sys.modules
    assert "flow360.cloud.flow360_requests" not in sys.modules


def test_flow360_root_help_does_not_eagerly_import_sdk_command_modules(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    for module_name in (
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.project",
        "flow360.cli.assets",
        "flow360.cli.draft",
        "flow360.cli.folder",
        "flow360.cli.wait",
        "flow360.cloud.flow360_requests",
    ):
        sys.modules.pop(module_name, None)

    from flow360.cli import flow360  # pylint: disable=import-outside-toplevel,import-error

    result = CliRunner().invoke(flow360, ["--help"])

    assert result.exit_code == 0
    assert "flow360.cli.project" not in sys.modules
    assert "flow360.cli.assets" not in sys.modules
    assert "flow360.cli.draft" not in sys.modules
    assert "flow360.cli.folder" not in sys.modules
    assert "flow360.cli.wait" not in sys.modules
    assert "flow360.cloud.flow360_requests" not in sys.modules


def test_asset_group_help_does_not_import_simulation_summary(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    for module_name in (
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.assets",
        "flow360.cli.simulation_summary",
        "flow360.component.simulation.simulation_params",
    ):
        sys.modules.pop(module_name, None)

    from flow360.cli import flow360  # pylint: disable=import-outside-toplevel,import-error

    result = CliRunner().invoke(flow360, ["case", "--help"])

    assert result.exit_code == 0
    assert "flow360.cli.assets" in sys.modules
    assert "flow360.cli.simulation_summary" not in sys.modules
    assert "flow360.component.simulation.simulation_params" not in sys.modules


def test_sdk_configure_helper_does_not_import_cli_modules(monkeypatch, tmp_path):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    config_path = tmp_path / "config.toml"

    for module_name in ("flow360.cli", "flow360.cli.app", "flow360.cli.api_set_func"):
        sys.modules.pop(module_name, None)

    import flow360.user_config as user_config  # pylint: disable=import-outside-toplevel,import-error

    monkeypatch.setattr(user_config, "config_file", str(config_path))

    user_config.configure_apikey("test-key", environment="dev", profile="default")

    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "test-key"
    assert "flow360.cli.app" not in sys.modules
    assert "flow360.cli.api_set_func" not in sys.modules


def test_accessing_flow360_configure_does_not_import_cli_app(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    for module_name in (
        "flow360",
        "flow360._api",
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.api_set_func",
    ):
        sys.modules.pop(module_name, None)

    import flow360  # pylint: disable=import-outside-toplevel,import-error

    configure = flow360.configure

    assert callable(configure)
    assert "flow360.cli.app" not in sys.modules
    assert "flow360.cli.api_set_func" not in sys.modules
