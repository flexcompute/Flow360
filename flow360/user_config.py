"""
User Config
"""

import os

import toml

from .environment import prod
from .file_path import flow360_dir
from .log import log

config_file = os.path.join(flow360_dir, "config.toml")
DEFAULT_PROFILE = "default"


class BasicUserConfig:
    """
    Basic User Configuration.
    """

    def __init__(self):
        self._read_config()
        self.set_profile(DEFAULT_PROFILE)
        self._check_env_profile()
        self._apikey = None
        self._check_env_apikey()
        self._do_validation = True
        self._suppress_submit_warning = None

    def _check_env_profile(self):
        simcloud_profile = os.environ.get("SIMCLOUD_PROFILE", None)
        if simcloud_profile is not None and simcloud_profile != self.profile:
            log.info(f"Found env variable SIMCLOUD_PROFILE={simcloud_profile}")
            self.set_profile(simcloud_profile)

    def _check_env_apikey(self):
        apikey = os.environ.get("FLOW360_APIKEY", None)
        if self._apikey != apikey:
            log.info("Found env variable FLOW360_APIKEY, using as apikey")
            self._apikey = apikey

    @property
    def profile(self):
        """profile"""
        return self._profile

    def set_profile(self, profile: str = DEFAULT_PROFILE):
        """set_profile

        Parameters
        ----------
        profile : str, optional
            profile to be used, eg. dev, default, by default "default"
        """
        self._profile = profile
        if profile != DEFAULT_PROFILE:
            log.info(f"Using profile={profile} for apikey")

    def _read_config(self):
        self.config = {}
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as file_handler:
                self.config = toml.loads(file_handler.read())

    def apikey(self, env):
        """get apikey

        Returns
        -------
        str
            apikey from config.toml file. If found env variable FLOW360_APIKEY, it will be returned
        """

        self._check_env_profile()
        self._check_env_apikey()
        if self._apikey is not None:
            return self._apikey
        # Check if environment-specific apikey exists
        key = self.config.get(self.profile, {})

        if env.name != prod.name:
            # By default the production environment is used.
            # If other environment is used, check if the key exists
            key = key.get(env.name, None)
        if key is None:
            log.warning(f"Cannot find api key associated with environment '{env.name}'.")
        return None if key is None else key.get("apikey", "")

    def suppress_submit_warning(self):
        """locally suppress submit warning"""
        self._suppress_submit_warning = True

    def show_submit_warning(self):
        """locally show submit warning"""
        self._suppress_submit_warning = False

    def is_suppress_submit_warning(self):
        """suppress submit warning config

        Returns
        -------
        bool
            whether to suppress submit warnings
        """
        if self._suppress_submit_warning is not None:
            return self._suppress_submit_warning
        return self.config.get("user", {}).get("config", {}).get("suppress_submit_warning", False)

    def cancel_local_submit_warning_settings(self):
        """cancel local submit warning settings"""
        self._suppress_submit_warning = None

    @property
    def do_validation(self):
        """for handling user side validation (pydantic)

        Returns
        -------
        bool
            whether to do user side validation
        """
        return self._do_validation

    def disable_validation(self):
        """disable user side validation (pydantic)"""
        self._do_validation = False

    def enable_validation(self):
        """enable user side validation (pydantic)"""
        self._do_validation = True


UserConfig = BasicUserConfig()
