import pytest
from unittest.mock import patch, MagicMock
import flow360.version_check
from flow360.version_check import *
import requests


def test_get_supported_server_versions():
    # Prepare mock data
    mock_response = MagicMock()
    mock_response.__iter__.return_value = iter(
        [{"version": "v1.0.0"}, {"version": "v2.0.3b5"}])
    # Mock the http_util.Http.get method
    with patch('flow360.version_check.http_util.Http.get') as mock_get:
        mock_get.return_value = mock_response
        # Test with a valid appName
        versions = flow360.version_check.get_supported_server_versions(
            "flow360-python-client-v2")
        # Add appropriate assertions based on the expected behavior
        assert versions == ["v1.0.0", "v2.0.3b5"]

        # Test with an HTTPError
        mock_get.side_effect = requests.exceptions.HTTPError()
        with pytest.raises(requests.exceptions.HTTPError):
            flow360.version_check.get_supported_server_versions(
                "invalid-appName")

        # Check if the expected error message is raised
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            flow360.version_check.get_supported_server_versions(
                "flow360-python-client-v2")
        assert str(exc_info.value) == 'Error in connecting server'

    # Test with no version
    mock_response.__iter__.return_value = iter(
        [])
    with patch('flow360.version_check.http_util.Http.get') as mock_get:
        with pytest.raises(RuntimeError) as exc_info:
            flow360.version_check.get_supported_server_versions(
                "flow360-python-client-v2")
        assert str(exc_info.value) == 'Error in fetching supported versions'


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
        "1.0.0a1"
    ]
    latest_version = supported_versions[0]

    # Test with supported appName and current version in supported versions
    current_version = "2.1.0"

    with patch('flow360.version_check.__version__', current_version):
        with patch('flow360.version_check.get_supported_server_versions', return_value=supported_versions):
            version_status, version = check_client_version(
                "flow360-python-client-v2")
            assert version_status == VersionSupported.YES
            assert str(version) == current_version

    # Test with supported appName and current version not in supported versions
    current_version = "2.2.0"
    with patch('flow360.version_check.__version__', current_version):
        with patch('flow360.version_check.get_supported_server_versions', return_value=supported_versions):
            version_status, version = check_client_version(
                "flow360-python-client-v2")
            assert version_status == VersionSupported.NO
            assert str(version) == current_version

    # Test with supported appName and current version that can be upgraded
    current_version = "2.0.0b1"
    with patch('flow360.version_check.__version__', current_version):
        with patch('flow360.version_check.get_supported_server_versions', return_value=supported_versions):
            version_status, version = check_client_version(
                "flow360-python-client-v2")
            assert version_status == VersionSupported.CAN_UPGRADE
            assert str(version) == latest_version

    # Test with solver version semantics
    supported_versions = ['release-22.3.4.0', 'release-22.1.1.0', 'release-0.3.0',
                          'beta-22.3.4.100', 'beta-22.1.3.0', 'beta-0.3.0', 'dummy-22.1.3.0', 'du-1m-m2y3-22.1.3.0']
    latest_version = supported_versions[0]
    # Test with supported appName and current version in supported versions
    current_version = 'release-22.3.4.0'

    with patch('flow360.version_check.__version__', current_version):
        with patch('flow360.version_check.get_supported_server_versions', return_value=supported_versions):
            version_status, version = check_client_version(
                "solver")
            assert version_status == VersionSupported.YES
            assert str(version) == current_version

    # Test with supported appName and current version not in supported versions
    current_version = 'beta-22.3.1.100'
    with patch('flow360.version_check.__version__', current_version):
        with patch('flow360.version_check.get_supported_server_versions', return_value=supported_versions):
            version_status, version = check_client_version(
                "solver")
            assert version_status == VersionSupported.NO
            assert str(version) == current_version

    # Test with supported appName and current version that can be upgraded
    current_version = 'release-22.1.1.0'
    with patch('flow360.version_check.__version__', current_version):
        with patch('flow360.version_check.get_supported_server_versions', return_value=supported_versions):
            version_status, version = check_client_version(
                "solver")
            assert version_status == VersionSupported.CAN_UPGRADE
            assert str(version) == latest_version

# def test_client_version_get_info(capsys):
#     # Test VersionSupported.NO
#     with patch('version_check.check_client_version') as mock_check_client_version:
#         mock_check_client_version.return_value = (VersionSupported.NO, "1.0")
#         client_version_get_info()
#         captured = capsys.readouterr()
#         assert "Your version of CLI (1.0) is no longer supported." in captured.out

#     # Test VersionSupported.CAN_UPGRADE
#     with patch('version_check.check_client_version') as mock_check_client_version:
#         mock_check_client_version.return_value = (
#             VersionSupported.CAN_UPGRADE, "2.0")
#         client_version_get_info()
#         captured = capsys.readouterr()
#         assert "New version of CLI (2.0) is now available." in captured.out


#     # Test VersionSupported.YES
#     with patch('version_check.check_client_version') as mock_check_client_version:
#         mock_check_client_version.return_value = (VersionSupported.YES, "1.0")
#         client_version_get_info()
#         captured = capsys.readouterr()
#         assert captured.out == ""
# Run the tests
pytest.main()
