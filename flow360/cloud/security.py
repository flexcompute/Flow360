"""
Security related functions.
"""

import flow360.user_config as user_config
from flow360.environment import Env


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """

    apikey = user_config.UserConfig.apikey(Env.current)
    return apikey
