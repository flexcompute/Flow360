import json
from threading import Thread
from time import sleep
from urllib.parse import parse_qs, urlparse

import requests
import toml
from click.testing import CliRunner
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import flow360.cli.app as app
import flow360.cli.auth as auth
import flow360.user_config as user_config
from flow360.cli import flow360
from flow360.environment import Env


def _post_callback_with_retry(
    callback_url: str, payload: dict, attempts: int = 50, delay: float = 0.1
):
    last_error = None
    for _ in range(attempts):
        try:
            response = requests.post(callback_url, json=payload, timeout=5)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            sleep(delay)

    if last_error is not None:
        raise last_error


def _post_form_callback_with_retry(
    callback_url: str, payload: dict, attempts: int = 50, delay: float = 0.1
):
    last_error = None
    for _ in range(attempts):
        try:
            response = requests.post(callback_url, data=payload, timeout=5)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            sleep(delay)

    if last_error is not None:
        raise last_error


def _get_callback_with_retry(
    callback_url: str, payload: dict, attempts: int = 50, delay: float = 0.1
):
    last_error = None
    for _ in range(attempts):
        try:
            response = requests.get(callback_url, params=payload, timeout=5)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            sleep(delay)

    if last_error is not None:
        raise last_error


def _encrypt_callback_payload(public_key: str, payload: dict) -> tuple[str, str, str]:
    public_key_bytes = auth._urlsafe_b64decode(public_key)  # pylint: disable=protected-access
    public_key_obj = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public_key_bytes)
    ephemeral_private_key = ec.generate_private_key(ec.SECP256R1())
    shared_secret = ephemeral_private_key.exchange(ec.ECDH(), public_key_obj)
    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=auth.CALLBACK_ENCRYPTION_INFO,
    ).derive(shared_secret)
    iv = b"0123456789ab"
    plaintext = json.dumps(payload).encode("utf-8")
    ciphertext = AESGCM(aes_key).encrypt(iv, plaintext, None)
    ephemeral_public_key = ephemeral_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return (
        auth._urlsafe_b64encode(ciphertext),  # pylint: disable=protected-access
        auth._urlsafe_b64encode(ephemeral_public_key),  # pylint: disable=protected-access
        auth._urlsafe_b64encode(iv),  # pylint: disable=protected-access
    )


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
        _post_callback_with_retry(
            "http://127.0.0.1:8765/callback",
            {"state": "state123", "apikey": "dev-manual-key", "email": "dev@example.com"},
        )

    Thread(target=complete_manual_login, daemon=True).start()

    result = runner.invoke(flow360, ["login", "--dev"])

    assert result.exit_code == 0
    assert "Starting local login server on http://127.0.0.1:8765/callback." in result.output
    assert "flow360.dev-simulation.cloud/account/cli-login" in result.output
    assert (
        "Could not open your browser automatically. Navigate to this URL to authenticate:"
        in result.output
    )
    assert "Headless environment? Configure an API key manually with:" in result.output
    assert "flow360 configure --dev --apikey <apikey>" in result.output
    assert "Successfully logged in as dev@example.com" in result.output


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
                json={
                    "state": "state123",
                    "apikey": "dev-browser-key",
                    "email": "browser@example.com",
                },
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
                json={
                    "state": "state123",
                    "apikey": "dev-browser-key",
                    "email": "browser@example.com",
                },
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


def test_wait_for_login_accepts_form_encoded_callback(monkeypatch, tmp_path):
    config_path = _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")

    def fake_open(login_url):
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        callback_url = params["callback_url"][0]
        Thread(
            target=lambda: _post_form_callback_with_retry(
                callback_url,
                {
                    "state": "state123",
                    "apikey": "dev-form-key",
                    "email": "form@example.com",
                },
            ),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    result = auth.wait_for_login(environment=Env.dev, profile="default", timeout=5)

    assert result["status"] == "success"
    assert result["email"] == "form@example.com"
    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "dev-form-key"


def test_wait_for_login_accepts_query_callback(monkeypatch, tmp_path):
    config_path = _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")

    def fake_open(login_url):
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        callback_url = params["callback_url"][0]
        Thread(
            target=lambda: _get_callback_with_retry(
                callback_url,
                {
                    "state": "state123",
                    "apikey": "dev-query-key",
                    "email": "query@example.com",
                },
            ),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    result = auth.wait_for_login(environment=Env.dev, profile="default", timeout=5)

    assert result["status"] == "success"
    assert result["email"] == "query@example.com"
    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "dev-query-key"


def test_wait_for_login_accepts_encrypted_query_callback(monkeypatch, tmp_path):
    config_path = _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")

    def fake_open(login_url):
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        callback_url = params["callback_url"][0]
        callback_public_key = params["callback_public_key"][0]
        callback_encryption_algorithm = params["callback_encryption_algorithm"][0]

        assert callback_encryption_algorithm == auth.CALLBACK_ENCRYPTION_ALGORITHM
        encrypted_payload, ephemeral_public_key, encrypted_iv = _encrypt_callback_payload(
            callback_public_key,
            {
                "apikey": "dev-encrypted-key",
                "email": "encrypted@example.com",
            },
        )

        Thread(
            target=lambda: _get_callback_with_retry(
                callback_url,
                {
                    "state": "state123",
                    "payload": encrypted_payload,
                    "epk": ephemeral_public_key,
                    "iv": encrypted_iv,
                },
            ),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    result = auth.wait_for_login(environment=Env.dev, profile="default", timeout=5)

    assert result["status"] == "success"
    assert result["email"] == "encrypted@example.com"
    config = toml.loads(config_path.read_text())
    assert config["default"]["dev"]["apikey"] == "dev-encrypted-key"


def test_wait_for_login_rejects_invalid_encrypted_query_callback(monkeypatch, tmp_path):
    _patch_config_file(monkeypatch, tmp_path)
    monkeypatch.setattr(auth.secrets, "token_urlsafe", lambda _: "state123")
    response_holder = {}

    def fake_open(login_url):
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        callback_url = params["callback_url"][0]

        def send_invalid_callback():
            response_holder["response"] = requests.get(
                callback_url,
                params={
                    "state": "state123",
                    "payload": "invalid",
                    "epk": "invalid",
                    "iv": "invalid",
                },
                timeout=5,
            )

        Thread(target=send_invalid_callback, daemon=True).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    result = CliRunner().invoke(flow360, ["login", "--dev"])

    assert result.exit_code != 0
    assert "Encrypted login callback payload is invalid." in result.output
    assert response_holder["response"].status_code == 400
    assert "CLI login failed" in response_holder["response"].text
