"""
Security related functions.
"""

from ..environment import Env
from ..user_config import UserConfig


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """

    apikey = UserConfig.apikey(Env.current)
    return apikey
