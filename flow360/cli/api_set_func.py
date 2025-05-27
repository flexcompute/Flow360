"""Helper function to set up the API key for the user."""

from click.testing import CliRunner

from flow360 import user_config
from flow360.cli.app import configure
from flow360.log import log


def configure_caller(apikey: str):
    """Wrapper for configure to allow user using python scripts to configure flow360 with an apikey"""
    runner = CliRunner()

    # Construct CLI arguments as a list
<<<<<<< HEAD
    args = ["--apikey", apikey]
=======
    args = ["--apikey", apikey, "--profile", profile]

    if environment:
        if environment.lower() in ("dev", "uat"):
            args += ["--" + environment.lower()]
        elif environment.lower() == "prod":
            args += []
        else:
            args += ["--env", environment]
>>>>>>> 49c32499 ([FL-996] Added reloading after the configure() call to set up APIKey (#1091))

    # Invoke the `configure` command
    result = runner.invoke(configure, args)

    if result.exit_code != 0:
        log.error(result.output if result.output else str(result.exception))
    else:
        log.info("Configuration successful.")
        user_config.UserConfig = user_config.BasicUserConfig()  # Reload
