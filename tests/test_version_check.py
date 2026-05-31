import re
from unittest.mock import patch

import pytest
import requests

import flow360.version_check as vc
from flow360.exceptions import Flow360RuntimeError, Flow360WebError

from .utils import (
    empty_mock_webapi_data_version_check,
    generate_mock_webapi_data_version_check,
)


def test_get_supported_server_versions(mock_response):
    generate_mock_webapi_data_version_check()
    versions = vc.get_supported_server_versions()
    # Add appropriate assertions based on the expected behavior
    assert versions == ["1.0.0", "2.0.3b5"]

    # Test with an HTTPError
    with patch("flow360.version_check.http.portal_api_get") as mock_get:
        mock_get.side_effect = requests.exceptions.HTTPError()

        # Check if the expected error message is raised
        with pytest.raises(Flow360WebError) as exc_info:
            vc.get_supported_server_versions()
        assert str(exc_info.value) == "failed to retrieve the versions for flow360-python-client-v2"

    # Test with no version
    empty_mock_webapi_data_version_check()
    with pytest.raises(Flow360RuntimeError) as exc_info:
        vc.get_supported_server_versions()
    str(exc_info.value)
    assert (
        str(exc_info.value)
        == "Something went wrong when checking python client version. The supported"
        + " versions should not be empty for flow360-python-client-v2. If you see this"
        + " message again, contact support."
    )


def test_check_client_version():
    # Test with PEP440 version naming semantics
    supported_versions = [
        "2.1.0",
        "2.1.0b3",
        "2.1.0b2",
        "2.1.0b1",
        "2.0.0",
        "2.0.0rc1",
        "2.0.0b2",
        "2.0.0b1",
        "1.0.0",
        "1.0.0a1",
    ]
    latest_version = supported_versions[0]

    # Test with supported app_name and current version in supported versions
    current_version = "2.1.0"

    with patch("flow360.version_check.__version__", current_version):
        with patch(
            "flow360.version_check.get_supported_server_versions", return_value=supported_versions
        ):
            version_status, version = vc.check_client_version()
            assert version_status == vc.VersionSupported.YES
            assert str(version) == current_version

    # Test with supported app_name and current version not in supported versions
    current_version = "2.2.0"
    with patch("flow360.version_check.__version__", current_version):
        with patch(
            "flow360.version_check.get_supported_server_versions", return_value=supported_versions
        ):
            version_status, version = vc.check_client_version()
            assert version_status == vc.VersionSupported.NO
            assert str(version) == current_version

    # Test with supported app_name and current version that can be upgraded
    current_version = "2.0.0b1"
    with patch("flow360.version_check.__version__", current_version):
        with patch(
            "flow360.version_check.get_supported_server_versions", return_value=supported_versions
        ):
            version_status, version = vc.check_client_version()
            assert version_status == vc.VersionSupported.CAN_UPGRADE
            assert str(version) == latest_version


def test_client_version_get_info(capfd):
    # Test VersionSupported.NO
    with patch("flow360.version_check.check_client_version") as mock_check_client_version:
        mock_check_client_version.return_value = (vc.VersionSupported.NO, "1.0")
        vc.client_version_get_info()
        captured = capfd.readouterr()
        captured_out = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)
        expected_message = (
            "ERROR: This flow360 version is no longer supported. Please upgrade or contact support."
        )
        assert expected_message in " ".join(captured_out.split()).replace("\n", "")
    # Test VersionSupported.CAN_UPGRADE
    with patch("flow360.version_check.check_client_version") as mock_check_client_version:
        mock_check_client_version.return_value = (vc.VersionSupported.CAN_UPGRADE, "2.0")
        vc.client_version_get_info()
        captured = capfd.readouterr()
        captured_out = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)
        expected_message = "INFO: New version of CLI (2.0) is now available."
        assert expected_message in " ".join(captured_out.split()).replace("\n", "")

    # Test VersionSupported.YES
    with patch("flow360.version_check.check_client_version") as mock_check_client_version:
        mock_check_client_version.return_value = (vc.VersionSupported.YES, "1.0")
        vc.client_version_get_info()
        captured = capfd.readouterr()
        captured_out = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)
        assert captured_out == ""


# Run the tests
if __name__ == "__main__":
    pytest.main()
