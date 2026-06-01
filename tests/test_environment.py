from flow360 import Env
from flow360.cloud.rest_api import RestApi
from flow360.environment import EnvironmentConfig


def test_version():
    Env.dev.active()
    print(Env.current)
    assert Env.current.name == "dev"
    Env.prod.active()


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
