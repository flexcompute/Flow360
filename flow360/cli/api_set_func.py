"""Helper function to set up the API key for the user."""

from click.testing import CliRunner

from flow360 import user_config
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
        if environment.lower() in ("dev", "uat"):
            args += ["--" + environment.lower()]
        elif environment.lower() == "prod":
            args += []
        else:
            args += ["--env", environment]

    # Invoke the `configure` command
    result = runner.invoke(configure, args)

    if result.exit_code != 0:
        log.error(result.output if result.output else str(result.exception))
    else:
        log.info("Configuration successful.")
        user_config.UserConfig = user_config.BasicUserConfig()  # Reload
