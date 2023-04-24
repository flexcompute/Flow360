"""
Environment Setup
"""

from pydantic import BaseModel

from .user_config import UserConfig


class EnvironmentConfig(BaseModel):
    """
    Basic Configuration for definition environment.
    """

    name: str
    web_api_endpoint: str
    aws_region: str
    apikey_profile: str

    def active(self):
        """
        Activate the particular environment.
        :return:
        """
        Env.set_current(self)
        UserConfig.set_profile(self.apikey_profile)

    def get_real_url(self, path: str):
        """
        Get the real url for the particular environment.
        :param path:
        :return:
        """
        return "/".join([self.web_api_endpoint, path])


dev = EnvironmentConfig(
    name="dev",
    web_api_endpoint="https://flow360-api.dev-simulation.cloud",
    aws_region="us-east-1",
    apikey_profile="dev",
)

uat = EnvironmentConfig(
    name="uat",
    web_api_endpoint="https://uat-flow360-api.simulation.cloud",
    aws_region="us-gov-west-1",
    apikey_profile="default",
)

prod = EnvironmentConfig(
    name="prod",
    web_api_endpoint="https://flow360-api.simulation.cloud",
    aws_region="us-gov-west-1",
    apikey_profile="default",
)


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
        self._impersonate = None
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

    @property
    def impersonate(self):
        """
        Get the impersonate environment.
        :return:
        """
        return self._impersonate

    @impersonate.setter
    def impersonate(self, value):
        self._impersonate = value

    @impersonate.deleter
    def impersonate(self):
        self._impersonate = None


Env = Environment()
