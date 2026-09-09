import gzip
import inspect
import json
import logging

import pytest

from flow360.cloud.http_util import Http, api_key_auth
from flow360.cloud.request_compression import BYTES_PER_MB
from flow360.environment import Env, EnvironmentConfig
from flow360.exceptions import (
    Flow360AuthorisationError,
    Flow360WebError,
    Flow360WebTimeoutError,
)


class _Response:
    status_code = 200

    @staticmethod
    def json():
        return {"data": {"ok": True}}


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url=None, **kwargs):
        self.calls.append(("get", url, kwargs))
        return _Response()

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return _Response()

    def put(self, url, **kwargs):
        self.calls.append(("put", url, kwargs))
        return _Response()

    def delete(self, url, **kwargs):
        self.calls.append(("delete", url, kwargs))
        return _Response()

    def patch(self, url, **kwargs):
        self.calls.append(("patch", url, kwargs))
        return _Response()


def _environment(name, api_endpoint, portal_endpoint="https://portal.example.test"):
    return EnvironmentConfig(
        name=name,
        domain="example.test",
        web_api_endpoint=api_endpoint,
        web_url="https://web.example.test",
        portal_web_api_endpoint=portal_endpoint,
    )


def test_normal_api_methods_use_absolute_urls_without_active_environment(monkeypatch):
    session = _Session()
    client = Http(session)

    monkeypatch.setattr(Env, "_current", object())

    assert client.get("https://api.example.test/v2/projects") == {"ok": True}
    assert session.calls == [
        (
            "get",
            "https://api.example.test/v2/projects",
            {"json": None, "params": None, "auth": api_key_auth},
        )
    ]


def test_normal_api_methods_have_no_environment_contract():
    client = Http(_Session())

    assert not hasattr(client, "_environment_provider")
    assert not hasattr(client, "environment_provider")
    assert not hasattr(client, "_api_url")

    for method_name in ("get", "post", "put", "delete", "patch"):
        parameters = inspect.signature(getattr(client, method_name)).parameters
        assert next(iter(parameters)) == "url"
        assert "environment" not in parameters
        assert "environment_provider" not in parameters


def test_normal_api_methods_pass_auth_and_payload():
    session = _Session()
    client = Http(session)

    client.get("https://api.example.test/v2/items", json={"body": True}, params={"page": 1})
    client.post("https://api.example.test/v2/items", json={"name": "created"})
    client.put("https://api.example.test/v2/items/item-id", json={"name": "updated"})
    client.delete("https://api.example.test/v2/items/item-id")
    client.patch("https://api.example.test/v2/items/item-id", json={"name": "patched"})

    assert session.calls == [
        (
            "get",
            "https://api.example.test/v2/items",
            {"json": {"body": True}, "params": {"page": 1}, "auth": api_key_auth},
        ),
        (
            "post",
            "https://api.example.test/v2/items",
            {"json": {"name": "created"}, "auth": api_key_auth},
        ),
        (
            "put",
            "https://api.example.test/v2/items/item-id",
            {"json": {"name": "updated"}, "auth": api_key_auth},
        ),
        (
            "delete",
            "https://api.example.test/v2/items/item-id",
            {"auth": api_key_auth},
        ),
        (
            "patch",
            "https://api.example.test/v2/items/item-id",
            {"json": {"name": "patched"}, "auth": api_key_auth},
        ),
    ]


THRESHOLD_MB = 5


def _body_of_serialized_size(size):
    """Build a JSON body whose serialized form is exactly ``size`` bytes."""
    return {"data": "x" * (size - len(json.dumps({"data": ""})))}


def test_large_post_body_is_not_compressed_without_a_threshold():
    session = _Session()
    body = _body_of_serialized_size(THRESHOLD_MB * BYTES_PER_MB + 1)

    Http(session).post("https://api.example.test/v2/items", json=body)

    assert session.calls == [
        (
            "post",
            "https://api.example.test/v2/items",
            {"json": body, "auth": api_key_auth},
        )
    ]


def test_post_body_at_threshold_is_sent_uncompressed():
    session = _Session()
    body = _body_of_serialized_size(THRESHOLD_MB * BYTES_PER_MB)

    Http(session).post(
        "https://api.example.test/v2/items", json=body, compress_when_larger_than_mb=THRESHOLD_MB
    )

    assert session.calls == [
        (
            "post",
            "https://api.example.test/v2/items",
            {"json": body, "auth": api_key_auth},
        )
    ]


def test_post_body_above_threshold_is_gzipped():
    session = _Session()
    body = _body_of_serialized_size(THRESHOLD_MB * BYTES_PER_MB + 1)

    Http(session).post(
        "https://api.example.test/v2/items", json=body, compress_when_larger_than_mb=THRESHOLD_MB
    )

    (_, _, kwargs) = session.calls[0]
    assert kwargs["headers"] == {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    assert kwargs["auth"] is api_key_auth
    assert "json" not in kwargs
    assert json.loads(gzip.decompress(kwargs["data"])) == body
    assert len(kwargs["data"]) < THRESHOLD_MB * BYTES_PER_MB


def test_portal_api_get_uses_active_portal_environment(monkeypatch):
    environment = _environment(
        "portal",
        "https://api.example.test",
        portal_endpoint="https://portal-api.example.test",
    )
    session = _Session()
    client = Http(session)

    monkeypatch.setattr(Env, "_current", environment)

    assert client.portal_api_get("v2/folders", params={"limit": 2}) == {"ok": True}
    assert session.calls == [
        (
            "get",
            "https://portal-api.example.test/v2/folders",
            {"json": None, "params": {"limit": 2}, "auth": api_key_auth},
        )
    ]


class _ErrorResponse:
    def __init__(self, status_code, json_body=None, text="", json_raises=False):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("response is not JSON")
        return self._json_body


class _FixedResponseSession:
    def __init__(self, response):
        self._response = response

    def get(self, url=None, **kwargs):
        return self._response

    def post(self, url=None, **kwargs):
        return self._response

    def put(self, url=None, **kwargs):
        return self._response

    def delete(self, url=None, **kwargs):
        return self._response

    def patch(self, url=None, **kwargs):
        return self._response


def _client_returning(response):
    return Http(_FixedResponseSession(response))


def test_bad_request_surfaces_backend_error_message():
    client = _client_returning(_ErrorResponse(400, json_body={"error": "Name must not be empty"}))

    with pytest.raises(Flow360WebError, match="Bad request error: Name must not be empty"):
        client.patch("https://api.example.test/v2/projects/p-id", json={"name": ""})


def test_bad_request_without_error_key_falls_back_to_body_text():
    client = _client_returning(
        _ErrorResponse(400, json_body={"detail": "something"}, text="raw-400-body")
    )

    with pytest.raises(Flow360WebError, match="Bad request error: raw-400-body"):
        client.get("https://api.example.test/v2/projects")


def test_unexpected_status_surfaces_backend_error_message():
    client = _client_returning(
        _ErrorResponse(409, json_body={"error": "Item size has exceeded the limit"})
    )

    with pytest.raises(
        Flow360WebError,
        match="Unexpected response error: 409: Item size has exceeded the limit",
    ):
        client.patch("https://api.example.test/v2/projects/p-id", json={"description": "x"})


def test_unexpected_status_non_json_body_is_status_only():
    client = _client_returning(
        _ErrorResponse(502, json_raises=True, text="<html>Bad Gateway</html>")
    )

    with pytest.raises(Flow360WebError) as exc:
        client.get("https://api.example.test/v2/projects")

    assert str(exc.value).endswith("Unexpected response error: 502")


@pytest.mark.parametrize("status_code", [408, 504])
def test_timeout_status_raises_without_logging(caplog, status_code):
    # A gateway that stops waiting says nothing about what the server did with the request,
    # so it must not reach the user as an error — not even through the log. The submit path
    # resolves it against the cloud instead.
    client = _client_returning(
        _ErrorResponse(status_code, json_raises=True, text="<html>Gateway Timeout</html>")
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Flow360WebTimeoutError):
            client.post("https://api.example.test/v2/drafts/dft-id/run", json={})

    assert caplog.records == []
    # Every existing submit-path handler catches Flow360WebError; the quiet variant has to
    # keep landing there.
    assert issubclass(Flow360WebTimeoutError, Flow360WebError)


def test_other_server_errors_still_announce_themselves(caplog):
    client = _client_returning(_ErrorResponse(500, json_body={"error": "boom"}))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Flow360WebError):
            client.post("https://api.example.test/v2/drafts/dft-id/run", json={})

    assert [record.levelno for record in caplog.records] == [logging.ERROR]


def test_unauthorized_error_surfaces_server_response():
    server_message = "Unauthorized:None account associate with this API Key."
    client = _client_returning(_ErrorResponse(401, text=server_message))

    with pytest.raises(Flow360AuthorisationError, match=server_message):
        client.post("https://api.example.test/v2/volume-meshes", json={})
