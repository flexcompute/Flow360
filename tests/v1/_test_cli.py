import toml
from click.testing import CliRunner

import flow360.cli.app as app
import flow360.user_config as user_config
from flow360.cli import flow360


def _patch_config_file(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(user_config, "config_file", str(config_path))
    monkeypatch.setattr(app, "config_file", str(config_path))
    return config_path


class TestClass:
    def test_no_configure(self, monkeypatch, tmp_path):
        config_path = _patch_config_file(monkeypatch, tmp_path)
        runner = CliRunner()

        result = runner.invoke(flow360, ["configure", "--apikey", "apikey"])

        assert result.exit_code == 0
        assert result.output == "done.\n"
        config = toml.loads(config_path.read_text())
        assert config.get("default", {}).get("apikey", "") == "apikey"

    def test_configure(self, monkeypatch, tmp_path):
        config_path = _patch_config_file(monkeypatch, tmp_path)
        config_path.write_text(toml.dumps({"default": {"apikey": "apikey"}}))
        runner = CliRunner()

        result = runner.invoke(flow360, ["configure", "--apikey", "apikey"])

        assert result.exit_code == 0
        assert result.output == "done.\n"
        config = toml.loads(config_path.read_text())
        assert config.get("default", {}).get("apikey", "") == "apikey"

    def test_logout_removes_default_apikey(self, monkeypatch, tmp_path):
        config_path = _patch_config_file(monkeypatch, tmp_path)
        config_path.write_text(toml.dumps({"default": {"apikey": "apikey"}}))
        runner = CliRunner()

        result = runner.invoke(flow360, ["logout"])

        assert result.exit_code == 0
        assert result.output == "Removed stored API key for profile 'default' in environment 'prod'.\n"
        config = toml.loads(config_path.read_text())
        assert "apikey" not in config.get("default", {})

    def test_logout_removes_dev_apikey(self, monkeypatch, tmp_path):
        config_path = _patch_config_file(monkeypatch, tmp_path)
        config_path.write_text(toml.dumps({"default": {"dev": {"apikey": "dev-key"}}}))
        runner = CliRunner()

        result = runner.invoke(flow360, ["logout", "--dev"])

        assert result.exit_code == 0
        assert result.output == "Removed stored API key for profile 'default' in environment 'dev'.\n"
        config = toml.loads(config_path.read_text())
        assert "dev" not in config.get("default", {})

    def test_logout_reports_missing_apikey(self, monkeypatch, tmp_path):
        _patch_config_file(monkeypatch, tmp_path)
        runner = CliRunner()

        result = runner.invoke(flow360, ["logout", "--dev"])

        assert result.exit_code == 0
        assert result.output == "No stored API key found for profile 'default' in environment 'dev'.\n"
