"""
http utils. Example:
http.get(url)
"""

import os
import sys
from functools import wraps
from typing import Optional

import requests

from ..cli.auth_guidance import build_missing_api_key_message
from ..environment import Env
from ..exceptions import (
    Flow360AuthorisationError,
    Flow360WebError,
    Flow360WebNotFoundError,
    Flow360WebTimeoutError,
)
from ..log import log
from ..user_config import UserConfig
from ..version import __version__
from ._tls import make_session
from .request_compression import GZIP_HEADERS, compress_json_body
from .security import api_key

# Statuses the gateway in front of the webAPI produces on its own when a call outlives
# its limit, before the server has said anything.
TIMEOUT_STATUS_CODES = frozenset({408, 504})


def get_user_agent():
    """Get the user agent the current environment."""
    return os.environ.get(
        "FLOW360_AGENT",
        f"Python-Client/{__version__}/"
        f"Python-Version/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def api_key_auth(request):
    """
    Set the authentication.
    :param request:
    :return:
    """
    key = api_key()
    if not key:
        raise Flow360AuthorisationError(
            build_missing_api_key_message(Env.current.name, UserConfig.profile)
        )
    request.headers["simcloud-api-key"] = key
    request.headers["flow360-python-version"] = __version__
    if Env.impersonate:
        request.headers["FLOW360ACCESSUSER"] = Env.impersonate
    request.headers["User-Agent"] = get_user_agent()
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

        log.debug(f"response: {resp}")

        if resp.status_code == 400:
            body = resp.json()
            raise Flow360WebError(
                f"Web {args[1]}: Bad request error: {body.get('error', resp.text)}",
                auxiliary_json=body,
            )

        if resp.status_code == 401:
            raise Flow360AuthorisationError(
                f"Web {args[1]}: Unauthorized: {resp.text} "
                f"(if your API key is invalid, check it on {Env.current.web_url} in account section)."
            )

        if resp.status_code == 404:
            raise Flow360WebNotFoundError(f"Web {args[1]}: Not found error: {resp.json()}")

        if resp.status_code in TIMEOUT_STATUS_CODES:
            # A gateway timeout body is an HTML page, not the usual error envelope,
            # and it says nothing about what the server did with the request.
            raise Flow360WebTimeoutError(f"Web {args[1]}: Timed out: {resp.status_code}")

        if resp.status_code == 200:
            try:
                result = resp.json()
                return result.get("data")
            except ValueError:
                # Handle the case where the response does not contain JSON data
                return None

        # Whitelist known 500 errors:
        if resp.text.count("credit has expired") or resp.text.count("credit is not enough"):
            # Note: Top import results in "json" redefinition error.
            import json  # pylint: disable=import-outside-toplevel

            error_dict = json.loads(resp.text)
            raise Flow360WebError(
                f"Error: {error_dict.get('error', error_dict)}",
            )

        try:
            parsed = resp.json()
            error_detail = parsed.get("error") if isinstance(parsed, dict) else None
        except ValueError:
            error_detail = None
        message = f"Web {args[1]}: Unexpected response error: {resp.status_code}"
        if error_detail:
            message += f": {error_detail}"
        raise Flow360WebError(message)

    return wrapper


class Http:
    """
    Http util class.
    """

    def __init__(self, session: requests.Session):
        self.session = session

    @http_interceptor
    def portal_api_get(self, path: str, json=None, params=None):
        """
        Get the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.get(
            url=Env.current.get_portal_real_url(path), json=json, params=params, auth=api_key_auth
        )

    @http_interceptor
    def get(self, url: str, json=None, params=None):
        """
        Get the resource.
        :param url:
        :param json:
        :return:
        """
        return self.session.get(url=url, json=json, params=params, auth=api_key_auth)

    def post_json_envelope(self, url: str, json=None) -> dict:
        """POST and return the full response envelope, not just its ``data`` field.

        The interceptor keeps only ``data``, which loses endpoints that report
        structured failures inside the envelope — draft validation returns its
        error list in ``detail`` with ``data`` null. Non-200 responses raise.
        """
        resp = self.session.post(url, json=json, auth=api_key_auth)
        if resp.status_code != 200:
            raise Flow360WebError(f"Web {url}: Unexpected response error: {resp.status_code}")
        return resp.json()

    @http_interceptor
    def post(self, url: str, json=None, compress_when_larger_than_mb: Optional[float] = None):
        """
        Create the resource.
        :param url:
        :param json:
        :param compress_when_larger_than_mb: gzip the body once it grows past this size.
            Leave as None except for endpoints known to accept "Content-Encoding: gzip".
        :return:
        """
        compressed = compress_json_body(json, compress_when_larger_than_mb)
        if compressed is not None:
            return self.session.post(url, data=compressed, headers=GZIP_HEADERS, auth=api_key_auth)

        return self.session.post(url, json=json, auth=api_key_auth)

    @http_interceptor
    def put(self, url: str, json):
        """
        Update the resource.
        :param url:
        :param json:
        :return:
        """
        return self.session.put(url, json=json, auth=api_key_auth)

    @http_interceptor
    def delete(self, url: str):
        """
        Delete the resource.
        :param url:
        :return:
        """
        return self.session.delete(url, auth=api_key_auth)

    @http_interceptor
    def patch(self, url: str, json=None):
        """
        Patch the resource.
        :param url:
        :param json:
        :return:
        """
        return self.session.patch(url, json=json, auth=api_key_auth)


http = Http(make_session())
