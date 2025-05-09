"""Helper function to set up the API key for the user."""

from click.testing import CliRunner

from flow360.cli.app import configure
from flow360.log import log


def configure_caller(apikey: str, environment: str = None, profile: str = "default") -> None:
    """
    Function interface for configuring the API key for flow360.

    Parameters:
        apikey (str): The API key to be configured.
        environment (str, optional): The environment to use. Default is to set for production environment.
        profile (str, optional): The profile name. Defaults to "default".

    Returns:
        None
    """
    runner = CliRunner()

    # Construct CLI arguments as a list
    args = ["--apikey", apikey, "--profile", profile]

    if environment:
        args += ["--env", environment]

    # Invoke the `configure` command
    result = runner.invoke(configure, args)

    if result.exit_code != 0:
        log.info("Error:" + result.output)
    else:
        log.info("Configuration successful.")
