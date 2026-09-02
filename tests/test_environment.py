from flow360 import Env
from flow360.environment import EnvironmentConfig, current_environment


def test_predefined_environment_activation(monkeypatch):
    monkeypatch.setattr(Env, "_current", Env.prod)

    Env.dev.active()
    assert Env.current is Env.dev

    Env.prod.active()
    assert Env.current is Env.prod


def test_load_returns_predefined_environment_objects():
    assert Env.load("dev") is Env.dev
    assert Env.load("uat") is Env.uat
    assert Env.load("prod") is Env.prod
    assert Env.load("preprod") is Env.preprod


def test_environment_config_active_sets_current(monkeypatch):
    monkeypatch.setattr(Env, "_current", Env.prod)
    environment = EnvironmentConfig(
        name="custom",
        domain="example.test",
        web_api_endpoint="https://api.example.test",
        web_url="https://web.example.test",
        portal_web_api_endpoint="https://portal-api.example.test",
        apikey_profile="custom",
    )

    environment.active()

    assert Env.current is environment


def test_current_environment_returns_active_environment(monkeypatch):
    environment = EnvironmentConfig(
        name="provider",
        domain="example.test",
        web_api_endpoint="https://api.example.test",
        web_url="https://web.example.test",
        portal_web_api_endpoint="https://portal-api.example.test",
        apikey_profile="provider",
    )

    monkeypatch.setattr(Env, "_current", Env.dev)
    assert current_environment() is Env.dev

    monkeypatch.setattr(Env, "_current", environment)
    assert current_environment() is environment


def test_from_on_premises_url_derives_all_endpoints():
    environment = EnvironmentConfig.from_on_premises_url(
        name="my_on_premises", base_url="http://localhost:80/"
    )

    assert environment.name == "my_on_premises"
    assert environment.web_api_endpoint == "http://localhost:80/flow360-api"
    assert environment.web_url == "http://localhost:80/flow360"
    assert environment.portal_web_api_endpoint == "http://localhost:80/flow360-portal-api"
    assert environment.s3_endpoint_url == "http://localhost:80/s3"


def test_from_on_premises_url_drops_path_from_web_ui_urls():
    for pasted in (
        "http://localhost:280/flow360/",
        "http://localhost:280/flow360",
        "http://localhost:280/flow360/workbench/prj-123",
    ):
        environment = EnvironmentConfig.from_on_premises_url(name="my_on_premises", base_url=pasted)
        assert environment.web_api_endpoint == "http://localhost:280/flow360-api"
        assert environment.web_url == "http://localhost:280/flow360"
        assert environment.portal_web_api_endpoint == "http://localhost:280/flow360-portal-api"
        assert environment.s3_endpoint_url == "http://localhost:280/s3"


def test_from_on_premises_url_assumes_http_without_scheme():
    environment = EnvironmentConfig.from_on_premises_url(
        name="my_on_premises", base_url="localhost:280"
    )
    assert environment.web_api_endpoint == "http://localhost:280/flow360-api"

    environment = EnvironmentConfig.from_on_premises_url(
        name="my_on_premises", base_url="nexus.example.com/flow360/"
    )
    assert environment.web_api_endpoint == "http://nexus.example.com/flow360-api"


def test_from_on_premises_url_s3_endpoint_override():
    environment = EnvironmentConfig.from_on_premises_url(
        name="my_on_premises",
        base_url="http://localhost",
        s3_endpoint_url="http://localhost:9000",
    )
    assert environment.s3_endpoint_url == "http://localhost:9000"

    environment = EnvironmentConfig.from_on_premises_url(
        name="my_on_premises", base_url="http://localhost", s3_endpoint_url=None
    )
    assert environment.s3_endpoint_url is None
