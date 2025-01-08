"""Helper function to set up the API key for the user."""

from click.testing import CliRunner

from flow360.cli.app import configure
from flow360.log import log


def configure_caller(apikey: str):
    """Wrapper for configure to allow user using python scripts to configure flow360 with an apikey"""
    runner = CliRunner()

    # Construct CLI arguments as a list
    args = ["--apikey", apikey]

    # Invoke the `configure` command
    result = runner.invoke(configure, args)

    if result.exit_code != 0:
        log.info("Error:" + result.output)
    else:
        log.info("Configuration successful.")
