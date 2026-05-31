import inspect

from flow360.cloud.http_util import Http, api_key_auth
from flow360.environment import Env, EnvironmentConfig


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
        apikey_profile="default",
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
