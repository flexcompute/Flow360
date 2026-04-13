from threading import Thread
from time import sleep
from urllib.parse import parse_qs, urlparse

import requests
import toml
from click.testing import CliRunner

import flow360.cli.auth as auth
import flow360.cli.app as app
import flow360.user_config as user_config
from flow360.cli import flow360
from flow360.environment import Env


def _patch_config_file(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(user_config, "config_file", str(config_path))
    monkeypatch.setattr(app, "config_file", str(config_path))
    return config_path


def test_configure_stores_dev_apikey(monkeypatch, tmp_path):
    config_path = _patch_config_file(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(flow360, ["configure", "--apikey", "dev-key", "--dev"])

    assert result.exit_code == 0
    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "dev-key"


def test_login_uses_dev_web_url_with_manual_fallback(monkeypatch, tmp_path):
    _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "_find_available_port", lambda host: 8765)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")
    monkeypatch.setattr(auth.webbrowser, "open", lambda _: False)
    runner = CliRunner()

    def complete_manual_login():
        sleep(0.2)
        requests.post(
            "http://127.0.0.1:8765/callback",
            json={"state": "state123", "apikey": "dev-manual-key", "email": "dev@example.com"},
            timeout=5,
        )

    Thread(target=complete_manual_login, daemon=True).start()

    result = runner.invoke(flow360, ["login", "--dev"])

    assert result.exit_code == 0
    assert "Starting local login server on http://127.0.0.1:8765/callback." in result.output
    assert "flow360.dev-simulation.cloud/account/cli-login" in result.output
    assert "Could not open your browser automatically. Navigate to this URL to authenticate:" in result.output
    assert "Successfully logged in as dev@example.com" in result.output


def test_login_local_uses_local_dev_frontend(monkeypatch, tmp_path):
    config_path = _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "_find_available_port", lambda host: 8765)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")
    monkeypatch.setattr(auth.webbrowser, "open", lambda _: False)
    runner = CliRunner()

    def complete_local_login():
        sleep(0.2)
        requests.post(
            "http://127.0.0.1:8765/callback",
            json={"state": "state123", "apikey": "local-dev-key", "email": "local@example.com"},
            timeout=5,
        )

    Thread(target=complete_local_login, daemon=True).start()

    result = runner.invoke(flow360, ["login", "--local"])

    assert result.exit_code == 0
    assert "Starting local login server on http://127.0.0.1:8765/callback." in result.output
    assert "local.dev-simulation.cloud:3000/account/cli-login" in result.output
    assert "Successfully logged in as local@example.com" in result.output
    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "local-dev-key"


def test_login_prints_fallback_message_when_browser_opens(monkeypatch, tmp_path):
    _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")

    def fake_open(login_url):
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        callback_url = params["callback_url"][0]
        Thread(
            target=lambda: requests.post(
                callback_url,
                json={"state": "state123", "apikey": "dev-browser-key", "email": "browser@example.com"},
                timeout=5,
            ),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)
    runner = CliRunner()

    result = runner.invoke(flow360, ["login", "--dev"])

    assert result.exit_code == 0
    assert "Starting local login server on " in result.output
    assert "flow360.dev-simulation.cloud/account/cli-login" in result.output
    assert "If your browser did not open, navigate to this URL to authenticate:" in result.output
    assert "Successfully logged in as browser@example.com" in result.output


def test_build_login_url_omits_prod_env():
    login_url = auth.build_login_url(
        environment=Env.prod,
        callback_url="http://127.0.0.1:8765/callback",
        state="state123",
        profile="default",
    )

    parsed = urlparse(login_url)
    params = parse_qs(parsed.query)

    assert parsed.netloc == "flow360.simulation.cloud"
    assert params["profile"] == ["default"]
    assert params["state"] == ["state123"]
    assert params["callback_url"] == ["http://127.0.0.1:8765/callback"]
    assert "env" not in params


def test_wait_for_login_stores_dev_apikey(monkeypatch, tmp_path):
    config_path = _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")

    def fake_open(login_url):
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        callback_url = params["callback_url"][0]
        Thread(
            target=lambda: requests.post(
                callback_url,
                json={"state": "state123", "apikey": "dev-browser-key", "email": "browser@example.com"},
                timeout=5,
            ),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    result = auth.wait_for_login(environment=Env.dev, profile="default", timeout=5)

    assert result["status"] == "success"
    assert result["email"] == "browser@example.com"
    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "dev-browser-key"
