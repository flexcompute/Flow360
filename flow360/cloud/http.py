"""
http utils.
"""
import requests
from requests.auth import AuthBase

from flow360 import Env
from flow360.cloud.security import api_key


# pylint: disable=too-few-public-methods
class APIKeyAuth(AuthBase):
    """
    http authentication for api key way.
    """

    def __init__(self):
        """
        Initialize the authentication.
        """

    def __call__(self, r):
        """
        Set the authentication.
        :param r:
        :return:
        """
        key = api_key()
        if not key:
            raise ValueError("API key not found, please set it by commandline: flow360 configure.")
        r.headers["simcloud-api-key"] = key
        return r


class Http:
    """
    Http util class.
    """

    def __init__(self, session: requests.Session):
        self.session = session

    def get(self, path: str, json=None):
        """
        Get the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.get(Env.current.get_real_url(path), auth=APIKeyAuth(), json=json)

    def post(self, path: str, json):
        """
        Create the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.post(Env.current.get_real_url(path), data=json, auth=APIKeyAuth())

    def put(self, path: str, json):
        """
        Update the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.put(Env.current.get_real_url(path), data=json, auth=APIKeyAuth())

    def delete(self, path: str):
        """
        Delete the resource.
        :param path:
        :return:
        """
        return self.session.delete(Env.current.get_real_url(path), auth=APIKeyAuth())


http = Http(requests.Session)
