"""
User Config
"""

import os
from typing import Optional

import toml

from .environment import prod
from .file_path import flow360_dir
from .log import log

config_file = os.path.join(flow360_dir, "config.toml")
DEFAULT_PROFILE = "default"
CONFIG_DIR_MODE = 0o700
CONFIG_FILE_MODE = 0o600


def _ensure_permissions(path: str, mode: int):
    """Best-effort permission hardening for local config paths."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


def ensure_config_dir():
    """Ensure the Flow360 config directory exists."""
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    _ensure_permissions(os.path.dirname(config_file), CONFIG_DIR_MODE)


def read_user_config():
    """Read the user config file if present."""
    if os.path.exists(config_file):
        _ensure_permissions(config_file, CONFIG_FILE_MODE)
        with open(config_file, encoding="utf-8") as file_handler:
            return toml.loads(file_handler.read())
    return {}


def write_user_config(config):
    """Write the user config file."""
    ensure_config_dir()
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    file_descriptor = os.open(config_file, flags, CONFIG_FILE_MODE)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handler:
        file_handler.write(toml.dumps(config))
    _ensure_permissions(config_file, CONFIG_FILE_MODE)


def store_apikey(
    apikey: str, profile: str = DEFAULT_PROFILE, environment_name: Optional[str] = None
):
    """Store an API key using the same config layout consumed by UserConfig."""
    config = read_user_config()

    if environment_name in (None, "", prod.name):
        entry = {profile: {"apikey": apikey}}
    else:
        entry = {profile: {environment_name: {"apikey": apikey}}}

    # Avoid importing CLI modules at import time because the wider package has lazy-import paths.
    from flow360.cli import dict_utils  # pylint: disable=import-outside-toplevel

    dict_utils.merge_overwrite(config, entry)
    write_user_config(config)
    return config


def delete_apikey(profile: str = DEFAULT_PROFILE, environment_name: Optional[str] = None):
    """Delete a stored API key for the selected profile/environment if present."""
    config = read_user_config()
    profile_config = config.get(profile)

    if not isinstance(profile_config, dict):
        return False, config

    removed = False
    if environment_name in (None, "", prod.name):
        removed = profile_config.pop("apikey", None) is not None
    else:
        env_config = profile_config.get(environment_name)
        if isinstance(env_config, dict):
            removed = env_config.pop("apikey", None) is not None
            if not env_config:
                profile_config.pop(environment_name, None)

    if not profile_config:
        config.pop(profile, None)

    if removed:
        write_user_config(config)
    return removed, config


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
        self.config = read_user_config()

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
