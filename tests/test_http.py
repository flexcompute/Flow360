import os
from os.path import expanduser

import pytest
import requests
from click.testing import CliRunner

from flow360.cloud.http import APIKeyAuth


def test_apikey_auth():
    runner = CliRunner()
    from flow360.cli import flow360

    result = runner.invoke(flow360, ["configure"], input="apikey")
    auth = APIKeyAuth()
    r = requests.Request()
    auth(r)
    assert r.headers["simcloud-api-key"] == "apikey"


def test_no_apikey_auth():
    auth = APIKeyAuth()
    r = requests.Request()
    if os.path.exists(f"{expanduser('~')}/.flow360/config.toml"):
        os.remove(f"{expanduser('~')}/.flow360/config.toml")
    with pytest.raises(
        ValueError, match="API key not found, please set it by commandline: flow360 configure."
    ):
        auth(r)
