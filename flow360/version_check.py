"""
This module provides utility functions for handling client versions.

Classes:
- VersionSupported: Enumeration representing the different states of version support.

Functions:
- get_supported_server_versions(None) -> List[str]: Fetches a list of supported versions from the server.
- check_client_version(None) -> Tuple[VersionSupported, Union[str, Version]]:
    Checks the current client version against the available versions.
- client_version_get_info(None) -> None: Logs information about the client version.

Constants:
- YES: VersionSupported enum value indicating that the version is supported.
- NO: VersionSupported enum value indicating that the version is no longer supported.
- CAN_UPGRADE: VersionSupported enum value indicating that an upgrade is available.

"""

import re
from enum import Enum
from typing import List, Tuple, Union

from packaging.version import Version
from requests import HTTPError

from flow360.cloud.http_util import http
from flow360.exceptions import Flow360RuntimeError

from .environment import FLOW360_SKIP_VERSION_CHECK
from .exceptions import Flow360WebError
from .log import log
from .version import __version__


class VersionSupported(Enum):
    """
    Enumeration representing the different states of version support.
    """

    YES = "yes"
    NO = "no"
    CAN_UPGRADE = "can upgrade"


def get_supported_server_versions() -> List[str]:
    """
    Fetch a list of supported versions from the server

    Returns:
        List[str]: A list of supported version strings
    """
    try:
        response = http.portal_api_get("versions?appName=flow360-python-client-v2")

    except HTTPError as error:
        raise Flow360WebError(
            "failed to retrieve the versions for flow360-python-client-v2"
        ) from error

    versions = [re.sub(r".+-", "", item["version"]) for item in response]

    if len(versions) == 0:
        raise Flow360RuntimeError(
            "Something went wrong when checking python client version."
            + " The supported versions should not be empty for flow360-python-client-v2."
            + " If you see this message again, contact support."
        )

    return versions


def check_client_version() -> Tuple[VersionSupported, Union[str, Version]]:
    """
    Check the current client version against the available versions

    Returns:
        Tuple[VersionSupported, Union[str, Version]]: A tuple containing the version status
        and the current or latest version
    """
    supported_versions = get_supported_server_versions()
    latest_version = supported_versions[0]
    current_version = __version__

    current_version = Version(current_version)
    is_supported = any(current_version == Version(v) for v in supported_versions)
    can_upgrade = current_version < Version(latest_version)

    if not is_supported:
        return VersionSupported.NO, current_version

    if can_upgrade:
        return VersionSupported.CAN_UPGRADE, latest_version

    return VersionSupported.YES, current_version


def client_version_get_info() -> None:
    """
    Check the current client version against the available versions

    Returns:
        Tuple[VersionSupported, Union[str, Version]]: A tuple containing the version status
        and the current or latest version
    """
    version_status, version = check_client_version()

    if version_status == VersionSupported.NO:
        log.error("This flow360 version is no longer supported. Please upgrade or contact support.")

    elif version_status == VersionSupported.CAN_UPGRADE:
        log.info(f"New version of CLI ({version}) is now available.")
    else:
        return

    msg = """
    To upgrade run:
        pip3 install -U flow360client

    """
    log.info(msg)


if not FLOW360_SKIP_VERSION_CHECK:
    client_version_get_info()
