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


def _merge_overwrite(old: dict, new: dict):
    """Deep-merge dictionaries while overwriting conflicts from `new`."""

    for key, value in new.items():
        if key in old and isinstance(old[key], dict) and isinstance(value, dict):
            _merge_overwrite(old[key], value)
        else:
            old[key] = value
    return old


def _normalize_storage_environment_name(environment: Optional[str]) -> Optional[str]:
    """Normalize environment names used for config storage."""

    if environment is None:
        return None

    normalized = environment.strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered == prod.name:
        return None
    if lowered in ("dev", "uat"):
        return lowered
    return normalized


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
    environment_name = _normalize_storage_environment_name(environment_name)

    if environment_name is None:
        entry = {profile: {"apikey": apikey}}
    else:
        entry = {profile: {environment_name: {"apikey": apikey}}}

    _merge_overwrite(config, entry)
    write_user_config(config)
    return config


def configure_apikey(
    apikey: str,
    environment: Optional[str] = None,
    profile: str = DEFAULT_PROFILE,
) -> None:
    """SDK-facing helper for storing an API key without going through the CLI app."""

    store_apikey(
        apikey,
        profile=profile,
        environment_name=environment,
    )
    reload_user_config()
    log.info("Configuration successful.")


def delete_apikey(profile: str = DEFAULT_PROFILE, environment_name: Optional[str] = None):
    """Delete a stored API key for the selected profile/environment if present."""
    config = read_user_config()
    environment_name = _normalize_storage_environment_name(environment_name)
    profile_config = config.get(profile)

    if not isinstance(profile_config, dict):
        return False, config

    removed = False
    if environment_name is None:
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
            log.debug(f"No api key configured for environment '{env.name}'.")
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


def reload_user_config():
    """Reload the shared user-config object in place when possible."""
    # pylint: disable=protected-access

    global UserConfig  # pylint: disable=global-statement

    if isinstance(UserConfig, BasicUserConfig):  # pylint: disable=used-before-assignment
        do_validation = UserConfig.do_validation
        suppress_submit_warning = UserConfig._suppress_submit_warning
        BasicUserConfig.__init__(UserConfig)
        UserConfig._do_validation = do_validation
        UserConfig._suppress_submit_warning = suppress_submit_warning
    else:
        UserConfig = BasicUserConfig()
    return UserConfig


UserConfig = BasicUserConfig()
