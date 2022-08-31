"""
http utils. Example:
http.get(path)
"""
import requests

from flow360 import Env
from flow360.cloud.security import api_key


def api_key_auth(request):
    """
    Set the authentication.
    :param request:
    :return:
    """
    key = api_key()
    if not key:
        raise ValueError("API key not found, please set it by commandline: flow360 configure.")
    request.headers["simcloud-api-key"] = key
    return request


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
        return self.session.get(url=Env.current.get_real_url(path), auth=api_key_auth, json=json)

    def post(self, path: str, json):
        """
        Create the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.post(Env.current.get_real_url(path), data=json, auth=api_key_auth)

    def put(self, path: str, json):
        """
        Update the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.put(Env.current.get_real_url(path), data=json, auth=api_key_auth)

    def delete(self, path: str):
        """
        Delete the resource.
        :param path:
        :return:
        """
        return self.session.delete(Env.current.get_real_url(path), auth=api_key_auth)


http = Http(requests.Session())
