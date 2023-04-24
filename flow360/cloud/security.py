"""
Security related functions.
"""
from ..user_config import UserConfig


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """

    apikey = UserConfig.apikey()
    return apikey
