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


def use_system_certs() -> bool:
    """
    Get the use_system_certs configuration option for the current environment
    :return:
    """
    setting = UserConfig.use_system_certs
    return setting
