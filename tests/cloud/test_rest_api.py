from types import SimpleNamespace

from flow360.cloud.rest_api import RestApi
from flow360.environment import Env, EnvironmentConfig, current_environment


class _HttpRecorder:
    def __init__(self):
        self.calls = []

    def get(self, path, json=None, params=None):
        self.calls.append(("get", path, {"json": json, "params": params}))
        return {"method": "get"}

    def post(self, path, json=None):
        self.calls.append(("post", path, {"json": json}))
        return {"method": "post"}

    def put(self, path, json=None):
        self.calls.append(("put", path, {"json": json}))
        return {"method": "put"}

    def delete(self, path):
        self.calls.append(("delete", path, {}))
        return {"method": "delete"}

    def patch(self, path, json=None):
        self.calls.append(("patch", path, {"json": json}))
        return {"method": "patch"}


def _environment(name, api_endpoint):
    return EnvironmentConfig(
        name=name,
        domain="example.test",
        web_api_endpoint=api_endpoint,
        web_url="https://web.example.test",
        portal_web_api_endpoint="https://portal-api.example.test",
        apikey_profile="default",
    )


def test_rest_api_builds_absolute_urls_from_environment_provider():
    http = _HttpRecorder()
    environment = _environment("test", "https://api.example.test")

    api = RestApi(
        "v2/projects",
        id="1234567890123456",
        environment_provider=lambda: environment,
        http_client=http,
    )

    assert api.get(method="tree", json={"body": True}, params={"page": 1}) == {"method": "get"}
    assert api.post({"name": "created"}, method="items") == {"method": "post"}
    assert api.put({"name": "updated"}, method="items") == {"method": "put"}
    assert api.delete(method="items") == {"method": "delete"}
    assert api.patch({"name": "patched"}, method="items") == {"method": "patch"}

    assert http.calls == [
        (
            "get",
            "https://api.example.test/v2/projects/1234567890123456/tree",
            {"json": {"body": True}, "params": {"page": 1}},
        ),
        (
            "post",
            "https://api.example.test/v2/projects/1234567890123456/items",
            {"json": {"name": "created"}},
        ),
        (
            "put",
            "https://api.example.test/v2/projects/1234567890123456/items",
            {"json": {"name": "updated"}},
        ),
        ("delete", "https://api.example.test/v2/projects/1234567890123456/items", {}),
        (
            "patch",
            "https://api.example.test/v2/projects/1234567890123456/items",
            {"json": {"name": "patched"}},
        ),
    ]


def test_rest_api_path_argument_overrides_composed_path():
    http = _HttpRecorder()
    environment = _environment("test", "https://api.example.test")

    api = RestApi(
        "v2/projects",
        id="1234567890123456",
        environment_provider=lambda: environment,
        http_client=http,
    )

    assert api.get(path="custom/path", method="ignored") == {"method": "get"}

    assert http.calls == [
        ("get", "https://api.example.test/custom/path", {"json": None, "params": None})
    ]


def test_rest_api_reads_provider_at_request_time(monkeypatch):
    first = _environment("first", "https://api.first.example.test")
    second = _environment("second", "https://api.second.example.test")
    http = _HttpRecorder()
    api = RestApi("v2/projects", environment_provider=current_environment, http_client=http)

    monkeypatch.setattr(Env, "_current", first)
    assert api.get(method="summary") == {"method": "get"}
    assert http.calls[-1] == (
        "get",
        "https://api.first.example.test/v2/projects/summary",
        {"json": None, "params": None},
    )

    monkeypatch.setattr(Env, "_current", second)
    assert api.get(method="summary") == {"method": "get"}
    assert http.calls[-1] == (
        "get",
        "https://api.second.example.test/v2/projects/summary",
        {"json": None, "params": None},
    )


def test_rest_api_uses_provider_current_value_for_same_instance():
    first = _environment("first", "https://api.first.example.test")
    second = _environment("second", "https://api.second.example.test")
    environment_module = SimpleNamespace(current=first)
    http = _HttpRecorder()
    api = RestApi(
        "physics/data",
        environment_provider=lambda: environment_module.current,
        http_client=http,
    )

    assert api.get(method="summary") == {"method": "get"}
    environment_module.current = second
    assert api.get(method="summary") == {"method": "get"}

    assert http.calls == [
        (
            "get",
            "https://api.first.example.test/physics/data/summary",
            {"json": None, "params": None},
        ),
        (
            "get",
            "https://api.second.example.test/physics/data/summary",
            {"json": None, "params": None},
        ),
    ]
