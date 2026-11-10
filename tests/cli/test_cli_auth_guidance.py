from click.testing import CliRunner

from flow360.cli import flow360
from flow360.exceptions import Flow360AuthorisationError


def test_project_list_missing_apikey_shows_clean_auth_guidance(monkeypatch):
    def raise_auth_error(*args, **kwargs):
        raise Flow360AuthorisationError(
            "\n".join(
                [
                    "No API key configured for env=dev, profile=default.",
                    "Authenticate with:",
                    "  flow360 login --dev",
                    "For headless or manual setup:",
                    "  flow360 configure --dev --apikey <apikey>",
                ]
            )
        )

    project_group = flow360.get_command(None, "project")
    monkeypatch.setattr(project_group.commands["list"], "callback", raise_auth_error)

    result = CliRunner().invoke(flow360, ["--dev", "project", "list"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "flow360 login --dev" in result.output
    assert "flow360 configure --dev --apikey <apikey>" in result.output
