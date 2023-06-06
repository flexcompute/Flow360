import sys
import re
from enum import Enum
from distutils.version import LooseVersion

from requests import HTTPError

from .version import __version__

from .httputils import portalApiGet


class VersionSupported(Enum):
    YES = 1
    NO = 2
    CAN_UPGRADE = 3


def get_supported_server_versions():
    try:
        response = portalApiGet("versions?appName=flow360-python-client")
    except HTTPError:
        raise HTTPError('Error in connecting server')

    versions = [re.sub(r".+-", "", item['version']) for item in response]

    if (len(versions) == 0):
        raise RuntimeError('Error in fetching supported versions')

    return versions


def check_client_version():
    supported_versions = get_supported_server_versions()
    latest_version = supported_versions[0]
    current_version = __version__

    is_supported = any([LooseVersion(current_version) == LooseVersion(v)
                        for v in supported_versions])

    if not is_supported:
        return VersionSupported.NO, current_version

    elif LooseVersion(current_version) < LooseVersion(latest_version):
        return VersionSupported.CAN_UPGRADE, latest_version

    else:
        return VersionSupported.YES, current_version


def client_version_get_info():
    version_status, version = check_client_version()

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


client_version_get_info()
