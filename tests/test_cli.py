import os.path
from os.path import expanduser

import toml
from click.testing import CliRunner

from flow360.cli import flow360

home = expanduser("~")


def test_no_configure():
    runner = CliRunner()
    if os.path.exists(f"{home}/.flow360/config.toml"):
        os.remove(f"{home}/.flow360/config.toml")
    result = runner.invoke(flow360, ["configure"], input="apikey")
    assert result.exit_code == 0
    assert result.output == "API Key[]: apikey\ndone.\n"
    with open(f"{home}/.flow360/config.toml", "r") as f:
        config = toml.loads(f.read())
        assert config.get("default", {}).get("apikey", "") == "apikey"


def test_no_configure():
    runner = CliRunner()
    result = runner.invoke(flow360, ["configure"], input="apikey")
    assert result.exit_code == 0
    assert result.output == "API Key[apikey]: apikey\ndone.\n"
    with open(f"{home}/.flow360/config.toml", "r") as f:
        config = toml.loads(f.read())
        assert config.get("default", {}).get("apikey", "") == "apikey"
