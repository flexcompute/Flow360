"""
Security related functions.
"""

from flow360 import user_config
from flow360.environment import Env


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """

    apikey = user_config.UserConfig.apikey(Env.current)
    return apikey
