from flow360 import Env
<<<<<<< HEAD
from flow360.environment import EnvironmentConfig, current_environment
=======
from flow360.cloud.rest_api import RestApi
from flow360.environment import EnvironmentConfig
>>>>>>> a55953b4 (Fix duplicate slash URL handling for 25.8 (#2051))


def test_predefined_environment_activation(monkeypatch):
    monkeypatch.setattr(Env, "_current", Env.prod)

    Env.dev.active()
    assert Env.current is Env.dev

    Env.prod.active()
<<<<<<< HEAD
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
=======


def test_environment_urls_normalize_path_slashes():
    env = EnvironmentConfig(
        name="test",
        domain="example.com",
        web_api_endpoint="https://api.example.com/",
        web_url="https://web.example.com/",
        portal_web_api_endpoint="https://portal.example.com/",
        apikey_profile="default",
    )

    for path in ("v2/folders", "/v2/folders"):
        assert env.get_real_url(path) == "https://api.example.com/v2/folders"
        assert env.get_portal_real_url(path) == "https://portal.example.com/v2/folders"
        assert env.get_web_real_url(path) == "https://web.example.com/v2/folders"


def test_rest_api_url_normalizes_with_environment():
    env = EnvironmentConfig(
        name="test",
        domain="example.com",
        web_api_endpoint="https://api.example.com",
        web_url="https://web.example.com",
        portal_web_api_endpoint="https://portal.example.com",
        apikey_profile="default",
    )

    assert env.get_real_url(RestApi("/v2/folders")._url(None)) == (
        "https://api.example.com/v2/folders"
    )
    assert env.get_real_url(RestApi("v2/folders")._url(None)) == (
        "https://api.example.com/v2/folders"
    )
>>>>>>> a55953b4 (Fix duplicate slash URL handling for 25.8 (#2051))
