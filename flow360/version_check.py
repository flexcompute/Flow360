"Client version check module"
import re
import sys
from enum import Enum

import requests
from packaging.version import Version
from requests import HTTPError

from .cloud import http_util
from .solver_version import Flow360Version as sv
from .version import __version__


class VersionSupported(Enum):
    """
    If version is supported
    """

    YES = 1
    NO = 2
    CAN_UPGRADE = 3


def get_supported_server_versions(app_name):
    """
    fetch list of supported versions
    """
    try:
        session = requests.Session()
        http = http_util.Http(session)
        response = http.get(f"versions?app_name={app_name}")
    except HTTPError as exc:
        raise HTTPError("Error in connecting server") from exc

    versions = [re.sub(r".+-", "", item["version"]) for item in response]

    if len(versions) == 0:
        raise RuntimeError("Error in fetching supported versions")

    return versions


def check_client_version(app_name):
    """
    app_name: the app_name for filtering the versions
    check the current version with available versions
    if the current version is no loger supported, return VersionSupported.NO, current_version
    if the current version can be upgraded, return VersionSupported.CAN_UPGRADE, latest_version
    otherwise, return VersionSupported.YES, current_version
    """
    supported_versions = get_supported_server_versions(app_name)
    latest_version = supported_versions[0]
    current_version = __version__
    if app_name == "flow360-python-client-v2":
        current_version = Version(current_version)
        is_supported = any(current_version == Version(v) for v in supported_versions)
        can_upgrade = current_version < Version(latest_version)
    else:
        current_version = sv(current_version)
        is_supported = any(current_version == sv(v) for v in supported_versions)
        can_upgrade = current_version < sv(latest_version)

    if not is_supported:
        return VersionSupported.NO, current_version

    if can_upgrade:
        return VersionSupported.CAN_UPGRADE, latest_version

    return VersionSupported.YES, current_version


def client_version_get_info(app_name):
    """
    Get information on if the client version is 1)supported 2)the latest version
    throw SystemExit 0 error if client version is not supported
    """
    version_status, version = check_client_version(app_name)

    if version_status == VersionSupported.NO:
        print(f"\nYour version of CLI ({version}) is no longer supported.")
    elif version_status == VersionSupported.CAN_UPGRADE:
        print(f"\nNew version of CLI ({version}) is now available.")
    else:
        return

    msg = """
    To upgrade run:
        pip3 install -U flow360client

    """
    print(msg)

    if version_status == VersionSupported.NO:
        sys.exit(0)
