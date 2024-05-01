import os
from os.path import expanduser

import pytest
import requests
from click.testing import CliRunner

from flow360.cloud.http_util import Http, SystemHttpsAdapter, api_key_auth
from flow360.exceptions import Flow360AuthorisationError


def test_apikey_auth():
    runner = CliRunner()
    from flow360.cli import flow360

    runner.invoke(flow360, ["configure"], input="apikey")
    r = requests.Request()
    api_key_auth(r)
    assert r.headers["simcloud-api-key"] == "apikey"


def test_no_apikey_auth():
    r = requests.Request()
    if os.path.exists(f"{expanduser('~')}/.flow360/config.toml"):
        os.remove(f"{expanduser('~')}/.flow360/config.toml")
    with pytest.raises(Flow360AuthorisationError, match="API key not found"):
        api_key_auth(r)


def test_request_to_public_target():
    session = requests.Session()
    http = Http(session)
    result = http.get("/health")
    assert result.status_code == 200


def test_request_to_public_target_os_certs():
    session = requests.Session()
    session.mount("https://", SystemHttpsAdapter())
    http = Http(session)
    result = http.get("/health")
    assert result.status_code == 200


def test_system_connect_via_https_adapter():
    session = requests.Session()
    session.mount("https://", SystemHttpsAdapter())
    result = session.get("https://flow360-api.simulation.cloud/health")
    assert result.status_code == 200
    data = result.json()
    assert data["healthEnv"] == "prod"
