import sys
import re
from enum import Enum
from . solver_version import Flow360Version as sv
from packaging.version import Version

from requests import HTTPError

from .version import __version__

from .cloud import http_util
import requests


class VersionSupported(Enum):
    YES = 1
    NO = 2
    CAN_UPGRADE = 3


def get_supported_server_versions(appName):
    """
    fetch list of supported versions
    """
    try:
        session = requests.Session()
        http = http_util.Http(session)
        response = http.get(f"versions?appName={appName}")
    except HTTPError:
        raise HTTPError('Error in connecting server')

    versions = [re.sub(r".+-", "", item['version']) for item in response]

    if (len(versions) == 0):
        raise RuntimeError('Error in fetching supported versions')

    return versions


def check_client_version(appName):
    """
    appName: the appName for filtering the versions
    check the current version with available versions
    if the current version is no loger supported, return VersionSupported.NO, current_version
    if the current version can be upgraded, return VersionSupported.CAN_UPGRADE, latest_version
    otherwise, return VersionSupported.YES, current_version
    """
    supported_versions = get_supported_server_versions(appName)
    latest_version = supported_versions[0]
    current_version = __version__
    if appName == "flow360-python-client-v2":
        current_version = Version(current_version)
        is_supported = any([current_version == Version(v)
                            for v in supported_versions])
        can_upgrade = current_version < Version(latest_version)
    else:
        current_version = sv(current_version)
        is_supported = any([current_version == sv(v)
                            for v in supported_versions])
        can_upgrade = current_version < sv(latest_version)

    if not is_supported:
        return VersionSupported.NO, current_version

    elif can_upgrade:
        return VersionSupported.CAN_UPGRADE, latest_version

    else:
        return VersionSupported.YES, current_version


def client_version_get_info(appName):
    """
    Get information on if the client version is 1)supported 2)the latest version
    throw SystemExit 0 error if client version is not supported
    """
    version_status, version = check_client_version(appName)

    if version_status == VersionSupported.NO:
        print("\nYour version of CLI ({}) is no longer supported.".format(version))
    elif version_status == VersionSupported.CAN_UPGRADE:
        print("\nNew version of CLI ({}) is now available.".format(version))
    else:
        return

    msg = """
    To upgrade run:
        pip3 install -U flow360client

    """
    print(msg)

    if version_status == VersionSupported.NO:
        sys.exit(0)
