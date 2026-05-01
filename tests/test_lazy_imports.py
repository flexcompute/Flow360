import ast
import sys
from pathlib import Path

import toml
from click.testing import CliRunner


def _unload_modules(monkeypatch, *module_names):
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def _load_api_all():
    api_source = Path(__file__).parents[1] / "flow360" / "_api.py"
    module_ast = ast.parse(api_source.read_text(encoding="utf-8"))
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return ast.literal_eval(node.value)
    raise AssertionError("flow360._api must define __all__")


def test_import_flow360_does_not_eagerly_import_heavy_dependencies(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    _unload_modules(monkeypatch, "flow360", "flow360._api", "pandas")

    import flow360  # pylint: disable=import-outside-toplevel,import-error

    assert "pandas" not in sys.modules
    assert "flow360._api" not in sys.modules
    assert "Env" in flow360.__all__


def test_import_flow360_cli_app_does_not_eagerly_import_sdk_command_modules(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    _unload_modules(
        monkeypatch,
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.project",
        "flow360.cli.assets",
        "flow360.cli.draft",
        "flow360.cli.folder",
        "flow360.cli.wait",
        "flow360.cloud.flow360_requests",
    )

    import flow360.cli.app  # pylint: disable=import-outside-toplevel,import-error,unused-import

    assert "flow360.cli.project" not in sys.modules
    assert "flow360.cli.assets" not in sys.modules
    assert "flow360.cli.draft" not in sys.modules
    assert "flow360.cli.folder" not in sys.modules
    assert "flow360.cli.wait" not in sys.modules
    assert "flow360.cloud.flow360_requests" not in sys.modules


def test_flow360_root_help_does_not_eagerly_import_sdk_command_modules(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    _unload_modules(
        monkeypatch,
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.project",
        "flow360.cli.assets",
        "flow360.cli.draft",
        "flow360.cli.folder",
        "flow360.cli.wait",
        "flow360.cloud.flow360_requests",
    )

    from flow360.cli import (
        flow360,  # pylint: disable=import-outside-toplevel,import-error
    )

    result = CliRunner().invoke(flow360, ["--help"])

    assert result.exit_code == 0
    assert "flow360.cli.project" not in sys.modules
    assert "flow360.cli.assets" not in sys.modules
    assert "flow360.cli.draft" not in sys.modules
    assert "flow360.cli.folder" not in sys.modules
    assert "flow360.cli.wait" not in sys.modules
    assert "flow360.cloud.flow360_requests" not in sys.modules


def test_sdk_configure_helper_does_not_import_cli_modules(monkeypatch, tmp_path):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    config_path = tmp_path / "config.toml"

    _unload_modules(monkeypatch, "flow360.cli", "flow360.cli.app", "flow360.cli.api_set_func")

    import flow360.user_config as user_config  # pylint: disable=import-outside-toplevel,import-error

    monkeypatch.setattr(user_config, "config_file", str(config_path))

    user_config.configure_apikey("test-key", environment="dev", profile="default")

    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "test-key"
    assert "flow360.cli.app" not in sys.modules
    assert "flow360.cli.api_set_func" not in sys.modules


def test_flow360_configure_is_exposed_without_importing_api_module(monkeypatch):
    monkeypatch.delenv("FLOW360_SUPPRESS_BETA_WARNING", raising=False)
    _unload_modules(
        monkeypatch,
        "flow360",
        "flow360._api",
        "flow360.cli",
        "flow360.cli.app",
        "flow360.cli.api_set_func",
    )

    import flow360  # pylint: disable=import-outside-toplevel,import-error

    assert "configure" in flow360.__all__
    assert "flow360._api" not in sys.modules
    assert "flow360.cli.app" not in sys.modules
    assert "flow360.cli.api_set_func" not in sys.modules


def test_flow360_version_check_legacy_lazy_attribute_does_not_import_api_module(monkeypatch):
    _unload_modules(monkeypatch, "flow360", "flow360._api", "flow360.version_check")

    import flow360  # pylint: disable=import-outside-toplevel,import-error

    assert flow360.version_check.__name__ == "flow360.version_check"
    assert "flow360._api" not in sys.modules


def test_flow360_stub_reexports_match_lazy_api_exports():
    stub_source = Path(__file__).parents[1] / "flow360" / "__init__.pyi"
    module_ast = ast.parse(stub_source.read_text(encoding="utf-8"))

    stub_api_reexports = set()
    has_version_check_reexport = False
    for node in module_ast.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "_api" and node.level == 1:
            for imported_name in node.names:
                assert imported_name.asname == imported_name.name
                stub_api_reexports.add(imported_name.name)
        if node.module is None and node.level == 1:
            has_version_check_reexport = any(
                imported_name.name == imported_name.asname == "version_check"
                for imported_name in node.names
            )

    assert stub_api_reexports == set(_load_api_all())
    assert has_version_check_reexport
