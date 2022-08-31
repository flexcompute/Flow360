import os
from os.path import expanduser

import pytest
import requests
from click.testing import CliRunner

from flow360 import Env
from flow360.cloud.http_util import api_key_auth, http


def test_apikey_auth():
    runner = CliRunner()
    from flow360.cli import flow360

    runner.invoke(flow360, ["configure"], input="apikey")
    r = requests.Request()
    api_key_auth(r)
    assert r.headers["simcloud-api-key"] == "apikey"
