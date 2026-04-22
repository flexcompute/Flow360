"""Authentication helpers for the Flow360 CLI."""

from __future__ import annotations

import html
import json
import secrets
import socket
import threading
import webbrowser
from base64 import urlsafe_b64decode, urlsafe_b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import flow360.user_config as user_config  # pylint: disable=consider-using-from-import
from flow360.environment import Env
from flow360.user_config import store_apikey

LOGIN_PATH = "account/cli-login"
CALLBACK_PATH = "/callback"
LOCAL_DEV_WEB_URL = "http://local.dev-simulation.cloud:3000"
DEV_WEB_URL = "https://flow360.dev-simulation.cloud"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_ENCRYPTION_ALGORITHM = "P-256-ECDH-AES-GCM-256"
CALLBACK_ENCRYPTION_INFO = b"flow360-cli-login-handoff"


class LoginError(RuntimeError):
    """Raised when CLI login fails."""


def resolve_target_environment(
    dev: bool = False,
    uat: bool = False,
    env: Optional[str] = None,
    local: bool = False,
):
    """Resolve the selected environment and validate conflicting CLI flags."""
    selected = [flag for flag, enabled in (("dev", dev), ("uat", uat), ("local", local)) if enabled]
    if env is not None:
        selected.append(env)

    if len(selected) > 1:
        raise ValueError("Use only one of --dev, --uat, --local, or --env.")

    if local:
        target = Env.dev
    elif dev:
        target = Env.dev
    elif uat:
        target = Env.uat
    elif env:
        target = Env.load(env)
    else:
        target = Env.prod

    storage_environment = None if target.name == Env.prod.name else target.name
    return target, storage_environment


def build_login_url(  # pylint: disable=too-many-arguments
    environment,
    callback_url: str,
    state: str,
    profile: str,
    callback_public_key: Optional[str] = None,
    callback_encryption_algorithm: Optional[str] = None,
    use_local_ui: bool = False,
) -> str:
    """Build the browser login URL for the selected environment."""
    query_params = {
        "callback_url": callback_url,
        "state": state,
        "profile": profile,
    }
    if environment.name != Env.prod.name:
        query_params["env"] = environment.name
    if callback_public_key:
        query_params["callback_public_key"] = callback_public_key
    if callback_encryption_algorithm:
        query_params["callback_encryption_algorithm"] = callback_encryption_algorithm

    query = urlencode(query_params)
    if use_local_ui:
        base_url = LOCAL_DEV_WEB_URL
    elif environment.name == Env.dev.name:
        base_url = DEV_WEB_URL
    else:
        base_url = environment.web_url
    return f"{base_url}/{LOGIN_PATH}?{query}"


def _find_available_port(host: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _urlsafe_b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    return urlsafe_b64decode(data + ("=" * (-len(data) % 4)))


def _generate_callback_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return private_key, _urlsafe_b64encode(public_key_bytes)


def _decrypt_callback_payload(
    encrypted_payload: str,
    encrypted_iv: str,
    ephemeral_public_key: str,
    private_key: ec.EllipticCurvePrivateKey,
) -> Dict[str, str]:
    try:
        ciphertext = _urlsafe_b64decode(encrypted_payload)
        iv = _urlsafe_b64decode(encrypted_iv)
        ephemeral_public_key_bytes = _urlsafe_b64decode(ephemeral_public_key)
        peer_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), ephemeral_public_key_bytes
        )
        shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
        aes_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=CALLBACK_ENCRYPTION_INFO,
        ).derive(shared_secret)
        plaintext = AESGCM(aes_key).decrypt(iv, ciphertext, None)
        payload = json.loads(plaintext.decode("utf-8"))
    except (InvalidTag, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LoginError("Encrypted login callback payload is invalid.") from error

    if not isinstance(payload, dict):
        raise LoginError("Encrypted login callback payload is invalid.")

    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, str)
    }


class _LoginCallbackServer(ThreadingHTTPServer):
    def __init__(self, server_address, *, expected_state: str, callback_private_key):
        super().__init__(server_address, _LoginCallbackHandler)
        self.callback_event = threading.Event()
        self.callback_params: Dict[str, str] = {}
        self.expected_state = expected_state
        self.callback_private_key = callback_private_key

    def process_callback_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Validate and normalize callback parameters before storing the API key."""
        if params.get("state") != self.expected_state:
            raise LoginError("Login callback state mismatch.")
        if "error" in params:
            raise LoginError(params["error"])
        if "payload" in params:
            encrypted_iv = params.get("iv")
            ephemeral_public_key = params.get("epk")
            if not encrypted_iv or not ephemeral_public_key:
                raise LoginError("Encrypted login callback payload is incomplete.")
            params = {
                **params,
                **_decrypt_callback_payload(
                    params["payload"],
                    encrypted_iv,
                    ephemeral_public_key,
                    self.callback_private_key,
                ),
            }
        if not params.get("apikey"):
            raise LoginError("Login callback did not include an API key.")
        return params


class _LoginCallbackHandler(BaseHTTPRequestHandler):
    server: _LoginCallbackServer

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _store_callback_params(self, params: Dict[str, str]):
        self.server.callback_params = params
        self.server.callback_event.set()

    def _send_browser_page(self, message: str, *, error: bool = False):
        title = "CLI login failed" if error else "Flow360 CLI connected"
        description = (
            "The local Flow360 CLI did not accept the login handoff."
            if error
            else "The local Flow360 CLI received your API key. You can return to your terminal and continue."
        )
        accent = "#d92d20" if error else "#12b76a"
        body = (
            "<!doctype html>"
            "<html lang='en'>"
            "<head>"
            "<meta charset='utf-8' />"
            "<meta name='viewport' content='width=device-width, initial-scale=1' />"
            "<title>{title}</title>"
            "<style>"
            ":root {{ color-scheme: light; }}"
            "* {{ box-sizing: border-box; }}"
            "body {{"
            "  margin: 0;"
            "  min-height: 100vh;"
            "  font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;"
            "  color: #101828;"
            "  background: radial-gradient(circle at top, rgba(18, 183, 106, 0.14) 0%, transparent 34%),"
            "              linear-gradient(180deg, #f5f7fa 0%, #ffffff 100%);"
            "}}"
            ".page {{"
            "  min-height: 100vh;"
            "  display: flex;"
            "  align-items: center;"
            "  justify-content: center;"
            "  padding: 32px 24px;"
            "}}"
            ".panel {{"
            "  width: min(560px, 100%);"
            "  padding: 32px;"
            "  border: 1px solid #d0d5dd;"
            "  border-radius: 12px;"
            "  background: #ffffff;"
            "  box-shadow: 0 16px 40px rgba(0, 39, 20, 0.08);"
            "}}"
            ".eyebrow {{"
            "  margin: 0 0 10px;"
            "  font-size: 13px;"
            "  font-weight: 700;"
            "  letter-spacing: 0.08em;"
            "  text-transform: uppercase;"
            "  color: #027a48;"
            "}}"
            ".title {{ margin: 0; font-size: 32px; line-height: 1.1; }}"
            ".description {{ margin: 12px 0 24px; color: #475467; font-size: 16px; line-height: 1.6; }}"
            ".alert {{"
            "  display: flex;"
            "  gap: 14px;"
            "  align-items: flex-start;"
            "  padding: 16px 18px;"
            "  border-radius: 8px;"
            "  background: #f8fafc;"
            "  border: 1px solid #eaecf0;"
            "}}"
            ".icon {{"
            "  width: 28px;"
            "  height: 28px;"
            "  border-radius: 999px;"
            "  flex: 0 0 28px;"
            "  margin-top: 1px;"
            "  background: {accent};"
            "  color: #ffffff;"
            "  display: inline-flex;"
            "  align-items: center;"
            "  justify-content: center;"
            "  font-size: 18px;"
            "  font-weight: 700;"
            "}}"
            ".alert-title {{ margin: 0 0 4px; font-size: 16px; font-weight: 700; }}"
            ".alert-copy {{ margin: 0; color: #475467; line-height: 1.6; }}"
            "</style>"
            "</head>"
            "<body>"
            "<div class='page'>"
            "<main class='panel'>"
            "<p class='eyebrow'>Flow360</p>"
            "<h1 class='title'>{title}</h1>"
            "<p class='description'>{description}</p>"
            "<section class='alert'>"
            "<div class='icon'>{icon}</div>"
            "<div>"
            "<p class='alert-title'>{alert_title}</p>"
            "<p class='alert-copy'>{message}</p>"
            "</div>"
            "</section>"
            "</main>"
            "</div>"
            "</body>"
            "</html>"
        ).format(
            title=html.escape(title),
            description=html.escape(description),
            alert_title=html.escape(title),
            message=html.escape(message),
            accent=accent,
            icon="!" if error else "✓",
        )
        encoded = body.encode("utf-8")

        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _parse_post_payload(self, raw_body: bytes):
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("Invalid JSON payload.") from error

            return {
                key: value
                for key, value in payload.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        if content_type.startswith("application/x-www-form-urlencoded"):
            try:
                decoded_body = raw_body.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("Invalid form payload.") from error

            return {key: values[-1] for key, values in parse_qs(decoded_body).items()}

        raise ValueError("Unsupported callback payload.")

    def do_OPTIONS(self):  # pylint: disable=invalid-name
        """Handle CORS preflight requests for the local callback endpoint."""
        parsed = urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_error(404)
            return

        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):  # pylint: disable=invalid-name
        """Handle browser redirects to the local callback endpoint."""
        parsed = urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_error(404)
            return

        raw_params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        try:
            params = self.server.process_callback_params(raw_params)
        except LoginError as error:
            self._store_callback_params({"error": str(error)})
            self.send_response(400)
            self._send_cors_headers()
            self._send_browser_page(str(error), error=True)
            return

        self._store_callback_params(params)

        message = params.get("message", "You can close this browser tab and return to the CLI.")
        self.send_response(200)
        self._send_cors_headers()
        self._send_browser_page(message)

    def do_POST(self):  # pylint: disable=invalid-name
        """Handle background JSON handoffs from the web login page."""
        parsed = urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            params = self._parse_post_payload(raw_body)
        except ValueError as error:
            self.send_response(400)
            self._send_cors_headers()
            self._send_browser_page(str(error), error=True)
            return

        try:
            validated_params = self.server.process_callback_params(params)
        except LoginError as error:
            self._store_callback_params({"error": str(error)})
            self.send_response(400)
            self._send_cors_headers()
            self._send_browser_page(str(error), error=True)
            return

        self._store_callback_params(validated_params)

        self.send_response(200)
        self._send_cors_headers()
        self._send_browser_page(
            validated_params.get("message", "You can close this browser tab and return to the CLI.")
        )

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        return


def wait_for_login(
    environment,
    profile: str,
    port: Optional[int] = None,
    timeout: int = 120,
    use_local_ui: bool = False,
    announce_login: Optional[Callable[[Dict[str, str]], None]] = None,
):  # pylint: disable=too-many-arguments,too-many-locals
    """Run the browser-based login flow and persist the resulting API key."""
    host = CALLBACK_HOST
    callback_port = port if port is not None else _find_available_port(host)
    callback_url = f"http://{host}:{callback_port}{CALLBACK_PATH}"
    state = secrets.token_urlsafe(24)
    callback_private_key, callback_public_key = _generate_callback_keypair()
    server = _LoginCallbackServer(
        (host, callback_port), expected_state=state, callback_private_key=callback_private_key
    )
    server.timeout = 0.2

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    login_url = build_login_url(
        environment,
        callback_url,
        state,
        profile,
        callback_public_key=callback_public_key,
        callback_encryption_algorithm=CALLBACK_ENCRYPTION_ALGORITHM,
        use_local_ui=use_local_ui,
    )

    try:
        try:
            opened = webbrowser.open(login_url)
        except webbrowser.Error:  # pragma: no cover - platform/browser dependent
            opened = False

        if announce_login is not None:
            announce_login(
                {
                    "login_url": login_url,
                    "callback_url": callback_url,
                    "browser_opened": "true" if opened else "false",
                    "environment": environment.name,
                    "profile": profile,
                }
            )

        if not server.callback_event.wait(timeout):
            raise LoginError(
                f"Timed out waiting for login callback after {timeout} seconds. "
                f"Retry with the same environment and open the printed URL manually if needed."
            )

        params = server.callback_params
        if "error" in params:
            raise LoginError(params["error"])

        apikey = params.get("apikey")
        if not apikey:
            raise LoginError("Login callback did not include an API key.")

        storage_environment = None if environment.name == Env.prod.name else environment.name
        store_apikey(apikey, profile=profile, environment_name=storage_environment)
        user_config.UserConfig = user_config.BasicUserConfig()
        return {
            "status": "success",
            "login_url": login_url,
            "callback_url": callback_url,
            "environment": environment.name,
            "profile": profile,
            "browser_opened": opened,
            "email": params.get("email"),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
