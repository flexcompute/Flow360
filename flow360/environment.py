"""
Environment Setup
"""
from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    """
    Basic Configuration for definition environment.
    """

    name: str
    web_api_endpoint: str

    def active(self):
        """
        Activate the particular environment.
        :return:
        """
        Env.set_current(self)


dev = EnvironmentConfig(name="dev", web_api_endpoint="https://dev-tidy3d-api.simulation.cloud")

uat = EnvironmentConfig(name="uat", web_api_endpoint="https://uat-tidy3d-api.simulation.cloud")

prod = EnvironmentConfig(name="prod", web_api_endpoint="https://prod-tidy3d-api.simulation.cloud")


class Environment:
    """
    Environment decorator for user interactive.
    For example:
        Env.dev.active()
        Env.current.name == "dev"
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        self._current = prod

    @property
    def current(self):
        """
        Get the current environment.
        :return: EnvironmentConfig
        """
        return self._current

    @property
    def dev(self):
        """
        Get the dev environment.
        :return:
        """
        return dev

    @property
    def uat(self):
        """
        Get the uat environment.
        :return:
        """
        return uat

    @property
    def prod(self):
        """
        Get the prod environment.
        :return:
        """
        return prod

    def set_current(self, config: EnvironmentConfig):
        """
        Set the current environment.
        :param config:
        :return:
        """
        self._current = config


Env = Environment()
