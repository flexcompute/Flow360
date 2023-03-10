"""
http utils. Example:
http.get(path)
"""
from functools import wraps

import requests

from ..environment import Env
from ..version import __version__
from ..exceptions import AuthorisationError, WebNotFoundError, WebError
from ..log import log

from .security import api_key


def api_key_auth(request):
    """
    Set the authentication.
    :param request:
    :return:
    """
    key = api_key(Env.current.apikey_profile)
    if not key:
        raise ValueError("API key not found, please set it by commandline: flow360 configure.")
    request.headers["simcloud-api-key"] = key
    request.headers["flow360-python-version"] = __version__
    return request


def http_interceptor(func):
    """
    Intercept the response and raise an exception if the status code is not 200.
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """A wrapper function"""

        # Extend some capabilities of func
        log.debug(f"call: {func.__name__}({args}, {kwargs})")

        resp = func(*args, **kwargs)

        if resp.status_code == 400:
            raise WebError(f"Web {args[1]}: Bad request error: {resp.json()['error']}")

        if resp.status_code == 401:
            raise AuthorisationError("Unauthorized.")

        if resp.status_code == 404:
            raise WebNotFoundError(f"Web {args[1]}: Not found error: {resp.json()}")

        if resp.status_code == 200:
            result = resp.json()
            return result.get("data")

        raise Exception(f"Web {args[1]}: Unexpected response error: {resp.status_code}")

    return wrapper


class Http:
    """
    Http util class.
    """

    def __init__(self, session: requests.Session):
        self.session = session

    @http_interceptor
    def get(self, path: str, json=None, params=None):
        """
        Get the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.get(
            url=Env.current.get_real_url(path), json=json, params=params, auth=api_key_auth
        )

    @http_interceptor
    def post(self, path: str, json=None):
        """
        Create the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.post(Env.current.get_real_url(path), json=json, auth=api_key_auth)

    @http_interceptor
    def put(self, path: str, json):
        """
        Update the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.put(Env.current.get_real_url(path), data=json, auth=api_key_auth)

    @http_interceptor
    def delete(self, path: str):
        """
        Delete the resource.
        :param path:
        :return:
        """
        return self.session.delete(Env.current.get_real_url(path), auth=api_key_auth)


http = Http(requests.Session())
