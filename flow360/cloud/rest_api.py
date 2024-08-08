"""
RestApi
"""

from ..component.utils import is_valid_uuid
from .http_util import http


class RestApi:
    """
    RestApi class
    """

    # pylint: disable=redefined-builtin
    def __init__(self, endpoint, id=None):
        is_valid_uuid(id, allow_none=True)
        self._id = id
        self._endpoint = endpoint

    def _url(self, method):
        url = f"{self._endpoint}"
        if self._id is not None:
            url += f"/{self._id}"
        if method is not None:
            url += f"/{method}"
        return url

    def get(self, path=None, method=None, json=None, params=None):
        """
        Resource get
        """
        return http.get(path or self._url(method), json=json, params=params)

    def post(self, json, path=None, method=None):
        """
        Resource post
        """
        return http.post(path or self._url(method), json=json)

    def put(self, json, path=None, method=None):
        """
        Resource put
        """
        return http.put(path or self._url(method), json=json)

    def delete(self, path=None, method=None):
        """
        Resource delete
        """
        return http.delete(path or self._url(method))

    def patch(self, json, path=None, method=None):
        """
        Resource patch
        """
        return http.patch(path or self._url(method), json=json)
