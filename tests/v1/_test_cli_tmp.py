import os.path
from os.path import expanduser

import shutil

import toml
from click.testing import CliRunner
import pytest

home = expanduser("~")


@pytest.fixture
def backup_and_restore_file():
    original_file = f"{home}/.flow360/config.toml"
    backup_file = f"{home}/.flow360/config.toml.bak"

    # Backup the original file
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"Backup of {original_file} created as {backup_file}")

    yield  # This allows the test to run

    # Restore the original file after the test
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, original_file)
        os.remove(backup_file)
        print(f"{original_file} restored from backup")


# @pytest.fixture(backup_and_restore_file)
def test_no_configure(backup_and_restore_file):
    if os.path.exists(f"{home}/.flow360/config.toml"):
        print("Deleting existing config file")
        os.remove(f"{home}/.flow360/config.toml")
    assert not os.path.exists(f"{home}/.flow360/config.toml")

    runner = CliRunner()

    from flow360.cli import flow360

    result = runner.invoke(flow360, ["configure"], input="apikey")
    assert result.exit_code == 0
    assert result.output == "API Key: apikey\ndone.\n"
    print(">>> result.output", result.output)
    with open(f"{home}/.flow360/config.toml") as f:
        config = toml.loads(f.read())
        print(">>> config", config)
        assert config.get("default", {}).get("apikey", "") == "apikey"


# @pytest.fixture(backup_and_restore_file)
def test_with_existing_configure(backup_and_restore_file):
    runner = CliRunner()
    from flow360.cli import flow360

    result = runner.invoke(flow360, ["configure"], input="apikey")
    assert result.exit_code == 0
    assert result.output == "API Key[apikey]: apikey\ndone.\n"
    with open(f"{home}/.flow360/config.toml") as f:
        config = toml.loads(f.read())
        assert config.get("default", {}).get("apikey", "") == "apikey"
