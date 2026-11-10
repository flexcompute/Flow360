"""Helper function to set up the API key for the user."""

from flow360.user_config import configure_apikey


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
    configure_apikey(apikey=apikey, environment=environment, profile=profile)
