"""
Environment Setup
"""

from __future__ import annotations

import os
from typing import Optional

import toml
from pydantic import BaseModel, ConfigDict

from .file_path import flow360_dir
from .log import log


class EnvironmentConfig(BaseModel):
    """
    Basic Configuration for definition environment.
    """

    name: str
    domain: str
    web_api_endpoint: str
    web_url: str
    apikey_profile: str
    portal_web_api_endpoint: Optional[str] = None
    s3_endpoint_url: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_domain(cls, name, domain, apikey_profile="default", **kwargs) -> EnvironmentConfig:
        """Create EnvironmentConfig using domain and populating to web_api_endpoint, web_url and portal_web_api_endpoint

        Parameters
        ----------
        name : str
            name, for example DEV
        domain : str
            domain, for example dev-simulation.cloud
        apikey_profile : str, optional
            profile to be used from flow360 config for apikey, by default 'default'

        Returns
        -------
        EnvironmentConfig
            completed EnvironmentConfig
        """
        if "aws_region" in kwargs:
            log.warning("`aws_region` is no longer a required argument. Input will be ignored.")

        env = cls(
            name=name,
            domain=domain,
            web_api_endpoint=f"https://flow360-api.{domain}",
            web_url=f"https://flow360.{domain}",
            portal_web_api_endpoint=f"https://portal-api.{domain}",
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

    @classmethod
    def from_config(cls, env_config_name: str):
        """
        Load specified environment configuration from config.toml.
        """
        config_file = os.path.join(flow360_dir, "config.toml")
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8", mode="r") as file_handler:
                config = toml.loads(file_handler.read())
                if "env_config" not in config:
                    raise ValueError("Cannot find any environment configuration in config file.")

                if env_config_name in config["env_config"]:
                    env_config = config["env_config"][env_config_name]
                    log.info(
                        f"Loaded environment configuration from config.toml for {env_config_name}."
                    )
                    return cls.model_validate(env_config)

                raise ValueError(
                    f"Environment configuration for `{env_config_name}` not found."
                    f" Available: {config['env_config'].keys()}."
                )
        else:
            raise FileNotFoundError("Failed to find the config file.")

    def save_config(self):
        """
        Save the configuration to the config file.
        """
        current_env_config = self.model_dump(mode="json")
        config_file = os.path.join(flow360_dir, "config.toml")
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8", mode="r") as file_handler:
                existing_config = toml.loads(file_handler.read())
                if "env_config" not in existing_config:
                    existing_config["env_config"] = {}
                existing_config["env_config"][self.name] = current_env_config
            with open(config_file, encoding="utf-8", mode="w") as file_handler:
                file_handler.write(toml.dumps(existing_config))
        else:
            # Create the config file if it doesn't exist.
            # This will not be triggered most likely.
            log.info("Creating config.toml since it does not exist.")
            with open(config_file, encoding="utf-8", mode="w") as file_handler:
                file_handler.write(toml.dumps({"env_config": {self.name: [current_env_config]}}))
        log.info("Saved environment configuration to config.toml.")


dev = EnvironmentConfig.from_domain(name="dev", domain="dev-simulation.cloud", apikey_profile="dev")
uat = EnvironmentConfig.from_domain(name="uat", domain="uat-simulation.cloud", apikey_profile="uat")
prod = EnvironmentConfig.from_domain(
    name="prod",
    domain="simulation.cloud",
)

preprod = EnvironmentConfig(
    name="preprod",
    domain="simulation.cloud",
    web_api_endpoint="https://preprod-flow360-api.simulation.cloud",
    web_url="https://preprod-flow360.simulation.cloud",
    portal_web_api_endpoint="https://preprod-portal-api.simulation.cloud",
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

    def load(self, /, env_config_name: str):
        """
        Load the environment configuration from config.toml.
        """
        if not isinstance(env_config_name, str):
            raise ValueError(
                f"The name of environment setting must be a string. Instead got {env_config_name}."
            )

        predefined_envs = (dev, uat, prod, preprod)
        for env in predefined_envs:
            if env_config_name == env.name:
                return env

        return EnvironmentConfig.from_config(env_config_name)

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
