import json
import os

from click.testing import CliRunner

from flow360.cli import flow360
from flow360.diagnostics import CheckResult, CheckStatus, DiagnosticReport

OK_REPORT = DiagnosticReport(checks=[CheckResult(name="install", status=CheckStatus.OK)])
FAIL_REPORT = DiagnosticReport(
    checks=[CheckResult(name="connectivity", status=CheckStatus.FAIL, detail="down")]
)


def test_diagnose_help_shows_options():
    runner = CliRunner()

    result = runner.invoke(flow360, ["diagnose", "--help"])

    assert result.exit_code == 0
    for option in ("--env", "--profile", "--json"):
        assert option in result.output
    for removed in ("--url", "--name", "--s3-endpoint", "--save"):
        assert removed not in result.output


def test_diagnose_defaults_to_active_environment(monkeypatch):
    captured = {}

    def fake_diagnose(env, *, source, print_report):
        captured["env"] = env
        captured["source"] = source
        return OK_REPORT

    monkeypatch.setattr("flow360.diagnostics.diagnose", fake_diagnose)
    runner = CliRunner()

    result = runner.invoke(flow360, ["diagnose"])

    assert result.exit_code == 0
    assert captured["env"].name == "prod"
    assert captured["source"] == "the active environment"


def test_diagnose_env_loads_named_environment(monkeypatch):
    captured = {}

    def fake_diagnose(env, *, source, print_report):
        captured["env"] = env
        captured["source"] = source
        return OK_REPORT

    monkeypatch.setattr("flow360.diagnostics.diagnose", fake_diagnose)
    runner = CliRunner()

    result = runner.invoke(flow360, ["diagnose", "--env", "dev"])

    assert result.exit_code == 0
    assert captured["env"].name == "dev"
    assert captured["source"] == "--env (config.toml)"


def test_diagnose_env_reports_unknown_environment(monkeypatch):
    monkeypatch.setattr(
        "flow360.environment.EnvironmentConfig.from_config",
        classmethod(lambda cls, name: (_ for _ in ()).throw(ValueError(f"`{name}` not found"))),
    )
    runner = CliRunner()

    result = runner.invoke(flow360, ["diagnose", "--env", "no_such_env"])

    assert result.exit_code == 1
    assert "no_such_env" in result.output


def test_diagnose_profile_is_set_during_run_and_restored(monkeypatch):
    import flow360.user_config as user_config

    observed = {}

    def fake_diagnose(env, *, source, print_report):
        observed["profile"] = user_config.UserConfig.profile
        return OK_REPORT

    monkeypatch.setattr("flow360.diagnostics.diagnose", fake_diagnose)
    monkeypatch.delenv("SIMCLOUD_PROFILE", raising=False)
    runner = CliRunner()

    result = runner.invoke(flow360, ["diagnose", "--profile", "secondary"])

    assert result.exit_code == 0
    assert observed["profile"] == "secondary"
    assert user_config.UserConfig.profile == "default"
    assert "SIMCLOUD_PROFILE" not in os.environ


def test_diagnose_json_output_and_failure_exit_code(monkeypatch):
    monkeypatch.setattr(
        "flow360.diagnostics.diagnose",
        lambda env, *, source, print_report: FAIL_REPORT,
    )
    runner = CliRunner()

    result = runner.invoke(flow360, ["diagnose", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["checks"][0]["name"] == "connectivity"
    assert payload["checks"][0]["status"] == "FAIL"
