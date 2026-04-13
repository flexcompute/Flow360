"""Authentication helpers for the Flow360 CLI."""

from __future__ import annotations

import html
import json
import secrets
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import flow360.user_config as user_config
from flow360.environment import Env
from flow360.user_config import store_apikey

LOGIN_PATH = "account/cli-login"
CALLBACK_PATH = "/callback"
LOCAL_DEV_WEB_URL = "http://local.dev-simulation.cloud:3000"


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


def build_login_url(
    environment,
    callback_url: str,
    state: str,
    profile: str,
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

    query = urlencode(query_params)
    base_url = LOCAL_DEV_WEB_URL if use_local_ui else environment.web_url
    return f"{base_url}/{LOGIN_PATH}?{query}"


def _find_available_port(host: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class _LoginCallbackServer(ThreadingHTTPServer):
    def __init__(self, server_address):
        super().__init__(server_address, _LoginCallbackHandler)
        self.callback_event = threading.Event()
        self.callback_params: Dict[str, str] = {}


class _LoginCallbackHandler(BaseHTTPRequestHandler):
    server: _LoginCallbackServer

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _store_callback_params(self, params: Dict[str, str]):
        self.server.callback_params = params
        self.server.callback_event.set()

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

        params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        self._store_callback_params(params)

        title = "Flow360 CLI Login"
        message = params.get("message", "You can close this browser tab and return to the CLI.")
        body = (
            "<html><head><title>{title}</title></head>"
            "<body><h1>{title}</h1><p>{message}</p></body></html>"
        ).format(title=html.escape(title), message=html.escape(message))
        encoded = body.encode("utf-8")

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

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
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_response(400)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"error":"Invalid JSON payload."}')
            return

        params = {
            key: value
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        self._store_callback_params(params)

        encoded = b'{"status":"ok"}'
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

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
    host = "127.0.0.1"
    callback_port = port if port is not None else _find_available_port(host)
    callback_url = f"http://{host}:{callback_port}{CALLBACK_PATH}"
    state = secrets.token_urlsafe(24)
    server = _LoginCallbackServer((host, callback_port))
    server.timeout = 0.2

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    login_url = build_login_url(
        environment, callback_url, state, profile, use_local_ui=use_local_ui
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
        if params.get("state") != state:
            raise LoginError("Login callback state mismatch.")
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
