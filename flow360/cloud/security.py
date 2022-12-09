"""
Security related functions.
"""
import os
from os.path import expanduser

import toml


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """

    apikey = os.environ.get("FLOW360_APIKEY", None)

    if apikey is None:
        profile = os.environ.get("SIMCLOUD_PROFILE", "default")
        if os.path.exists(f"{expanduser('~')}/.flow360/config.toml"):
            with open(
                f"{expanduser('~')}/.flow360/config.toml", "r", encoding="utf-8"
            ) as config_file:
                config = toml.loads(config_file.read())
                apikey = config.get(profile, {}).get("apikey", "")

    return apikey
