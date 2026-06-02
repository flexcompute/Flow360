"""
RestApi
"""

from .http_util import http
from .utils import is_valid_uuid


class RestApi:
    """
    RestApi class
    """

    # pylint: disable=redefined-builtin
    def __init__(self, endpoint, id=None, *, environment_provider, http_client=None):
        is_valid_uuid(id, allow_none=True)
        self._id = id
        self._endpoint = endpoint
        self._environment_provider = environment_provider
        self._http = http if http_client is None else http_client

    def _url(self, method):
        url = f"{self._endpoint}"
        if self._id is not None:
            url += f"/{self._id}"
        if method is not None:
            url += f"/{method}"
        return url

    @staticmethod
    def _api_url(path, environment):
        return environment.web_api_endpoint.rstrip("/") + "/" + path.lstrip("/")

    def get(self, path=None, method=None, json=None, params=None):
        """
        Resource get
        """
        path = path or self._url(method)
        return self._http.get(
            self._api_url(path, self._environment_provider()), json=json, params=params
        )

    def post(self, json, path=None, method=None):
        """
        Resource post
        """
        path = path or self._url(method)
        return self._http.post(self._api_url(path, self._environment_provider()), json=json)

    def put(self, json, path=None, method=None):
        """
        Resource put
        """
        path = path or self._url(method)
        return self._http.put(self._api_url(path, self._environment_provider()), json=json)

    def delete(self, path=None, method=None):
        """
        Resource delete
        """
        path = path or self._url(method)
        return self._http.delete(self._api_url(path, self._environment_provider()))

    def patch(self, json, path=None, method=None):
        """
        Resource patch
        """
        path = path or self._url(method)
        return self._http.patch(self._api_url(path, self._environment_provider()), json=json)
