"""
Environment Setup
"""

from __future__ import annotations

from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    """
    Basic Configuration for definition environment.
    """

    name: str
    domain: str
    web_api_endpoint: str
    web_url: str
    aws_region: str
    apikey_profile: str
    portal_web_api_endpoint: str = None

    @classmethod
    def from_domain(cls, name, domain, aws_region, apikey_profile="default") -> EnvironmentConfig:
        """Create EnvironmentConfig using domain and populating to web_api_endpoint, web_url and portal_web_api_endpoint

        Parameters
        ----------
        name : str
            name, for example DEV
        domain : str
            domain, for example dev-simulation.cloud
        aws_region : str
            aws region
        apikey_profile : str, optional
            profile to be used from flow360 config for apikey, by default 'default'

        Returns
        -------
        EnvironmentConfig
            completed EnvironmentConfig
        """
        env = cls(
            name=name,
            domain=domain,
            web_api_endpoint=f"https://flow360-api.{domain}",
            web_url=f"https://flow360.{domain}",
            portal_web_api_endpoint=f"https://portal-api.{domain}",
            aws_region=aws_region,
            apikey_profile=apikey_profile,
        )
        return env

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


dev = EnvironmentConfig.from_domain(
    name="dev", domain="dev-simulation.cloud", aws_region="us-east-1", apikey_profile="dev"
)
uat = EnvironmentConfig.from_domain(
    name="uat", domain="uat-simulation.cloud", aws_region="us-west-2"
)
prod = EnvironmentConfig.from_domain(
    name="prod", domain="simulation.cloud", aws_region="us-gov-west-1"
)

preprod = EnvironmentConfig(
    name="preprod",
    domain="simulation.cloud",
    web_api_endpoint="https://preprod-flow360-api.simulation.cloud",
    web_url="https://preprod-flow360.simulation.cloud",
    portal_web_api_endpoint="https://preprod-portal-api.simulation.cloud",
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

    @property
    def preprod(self):
        """
        Get the preprod environment.
        :return:
        """
        return preprod

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
