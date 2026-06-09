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
    )

    monkeypatch.setattr(Env, "_current", Env.dev)
    assert current_environment() is Env.dev

    monkeypatch.setattr(Env, "_current", environment)
    assert current_environment() is environment


def test_environment_urls_normalize_path_slashes():
    environment = EnvironmentConfig(
        name="provider",
        domain="example.test",
        web_api_endpoint="https://api.example.test/",
        web_url="https://web.example.test/",
        portal_web_api_endpoint="https://portal-api.example.test/",
    )

    for path in ("v2/folders", "/v2/folders"):
        assert environment.get_real_url(path) == "https://api.example.test/v2/folders"
        assert environment.get_portal_real_url(path) == (
            "https://portal-api.example.test/v2/folders"
        )
        assert environment.get_web_real_url(path) == "https://web.example.test/v2/folders"
