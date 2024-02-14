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
    web_url: str
    aws_region: str
    apikey_profile: str
    portal_web_api_endpoint: str = None

    def active(self):
        """
        Activate the particular environment.
        :return:
        """
        Env.set_current(self)

    def get_real_url(self, path: str):
        """
        Get the real url for the particular environment.
        :param path:
        :return:
        """
        return "/".join([self.web_api_endpoint, path])

    def get_portal_real_url(self, path: str):
        """
        Get the portal real url for the particular environment.
        :param path:
        :return:
        """
        return "/".join([self.portal_web_api_endpoint, path])

    def get_web_real_url(self, path: str):
        """
        Get the web real url for the particular environment.
        :param path:
        :return:
        """
        return "/".join([self.web_url, path])


dev = EnvironmentConfig(
    name="dev",
    web_api_endpoint="https://flow360-api.dev-simulation.cloud",
    web_url="https://flow360.dev-simulation.cloud",
    portal_web_api_endpoint="https://portal-api.dev-simulation.cloud",
    aws_region="us-east-1",
    apikey_profile="dev",
)

uat = EnvironmentConfig(
    name="uat",
    web_api_endpoint="https://uat-flow360-api.simulation.cloud",
    web_url="https://uat-flow360.simulation.cloud",
    portal_web_api_endpoint="https://uat-portal-api.simulation.cloud",
    aws_region="us-gov-west-1",
    apikey_profile="default",
)

prod = EnvironmentConfig(
    name="prod",
    web_api_endpoint="https://flow360-api.simulation.cloud",
    web_url="https://flow360.simulation.cloud",
    portal_web_api_endpoint="https://portal-api.simulation.cloud",
    aws_region="us-gov-west-1",
    apikey_profile="default",
)

FLOW360_SKIP_VERSION_CHECK = True


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
