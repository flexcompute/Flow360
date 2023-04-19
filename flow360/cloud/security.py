"""
Security related functions.
"""
from ..user_config import user_config


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """

    apikey = user_config.apikey()
    return apikey
