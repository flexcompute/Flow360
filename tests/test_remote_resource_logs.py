import os
from enum import Enum
from unittest.mock import Mock
import pytest

from flow360.component.resource_base import (
    Flow360Resource,
    Position,
    RemoteResourceLogs,
)


def create_file(file_name: str, to_file):
    content = (
        ("Info:" + file_name + "\n") * 2
        + ("Debug:" + file_name + "\n") * 2
        + ("Warning:" + file_name + "\n") * 2
        + ("Error:" + file_name + "\n") * 2
    ) * 3
    encoded_content = content.encode("utf-8")  # Encode the string content to bytes
    to_file.write(encoded_content)
    to_file.seek(0)


class TestRemoteResourceLogs:
    def setup_method(self):
        self.flow360_resource = Mock(spec=Flow360Resource)
        self.flow360_resource.get_download_file_list.return_value = ["file1.log", "file2.log"]
        self.flow360_resource.download_file.side_effect = lambda file_name, temp_file: create_file(
            file_name=file_name, to_file=temp_file
        )

        self.remote_logs = RemoteResourceLogs(self.flow360_resource)
        self.remote_logs.path = "file1.log"
        self.remote_logs.paths = ["file1.log"]

    def test_get_log_by_pos(self):
        # Mock the necessary methods and attributes
        # Test head
        log_lines = self.remote_logs._get_log_by_pos(Position.HEAD, num_lines=5)
        assert log_lines == ["Info:file1.log"] * 2 + ["Debug:file1.log"] * 2 + ["Warning:file1.log"]

        # Test tail with different number of lines
        self.remote_logs.paths.append("file2.log")
        print(self.remote_logs.paths)
        log_lines = self.remote_logs._get_log_by_pos(
            Position.TAIL, num_lines=2, file_name="file2.log"
        )
        assert log_lines == ["Error:file2.log"] * 2

        # Test all lines
        log_lines = self.remote_logs._get_log_by_pos(file_name="file1.log")
        assert (
            log_lines
            == (
                ["Info:file1.log"] * 2
                + ["Debug:file1.log"] * 2
                + ["Warning:file1.log"] * 2
                + ["Error:file1.log"] * 2
            )
            * 3
        )

    def test_get_log_by_level(self):
        # Mock the necessary methods and attributes
        self.remote_logs.path = "file1.log"

        # Test error
        log_lines = self.remote_logs._get_log_by_level("ERROR")
        assert log_lines == ["Error:file1.log"] * 6

        log_lines = self.remote_logs._get_log_by_level("WARNING")
        assert log_lines == (["Warning:file1.log"] * 2 + ["Error:file1.log"] * 2) * 3

        self.remote_logs.paths = ["file1.log", "file2.log"]
        log_lines = self.remote_logs._get_log_by_level("INFO", "file2.log")
        assert (
            log_lines
            == (["Info:file2.log"] * 2 + ["Warning:file2.log"] * 2 + ["Error:file2.log"] * 2) * 3
        )

        log_lines = self.remote_logs._get_log_by_level(file_name="file1.log")
        print(
            (
                ["Info:file1.log"] * 2
                + ["Debug:file1.log"] * 2
                + ["Warning:file1.log"] * 2
                + ["Error:file1.log"] * 2
            )
            * 3
        )
        assert (
            log_lines
            == (
                ["Info:file1.log"] * 2
                + ["Debug:file1.log"] * 2
                + ["Warning:file1.log"] * 2
                + ["Error:file1.log"] * 2
            )
            * 3
        )

    def test_head_tail_print(self):
        # Mock the necessary methods and attributes
        self.remote_logs._get_log_by_pos = Mock(return_value=["Line 1", "Line 2", "Line 3"])
        # self.remote_logs.clean = Mock(return_value=None)
        # Head
        self.remote_logs.head(num_lines=2)
        self.remote_logs._get_log_by_pos.assert_called_with(Position.HEAD, 2)
        # Tail
        self.remote_logs.tail(num_lines=2)
        self.remote_logs._get_log_by_pos.assert_called_with(Position.TAIL, 2)
        # Print
        self.remote_logs.print()
        self.remote_logs._get_log_by_pos.assert_called_with(Position.ALL)

    def test_errors_warnings_info(self):
        # Mock the necessary methods and attributes
        self.remote_logs._get_log_by_level = Mock(
            return_value=[
                "Error: Something went wrong",
                "Warning: This is a warning",
                "Info: Hey",
            ]
        )
        # self.remote_logs.clean = Mock(return_value=None)
        # Call the method under test
        self.remote_logs.errors()
        self.remote_logs._get_log_by_level.assert_called_with("ERROR")
        self.remote_logs.warnings()
        self.remote_logs._get_log_by_level.assert_called_with("WARNING")
        self.remote_logs.info()
        self.remote_logs._get_log_by_level.assert_called_with("INFO")

    def test_write_to_file(self):
        os.makedirs("./logs/", exist_ok=True)
        self.remote_logs._get_log_by_pos = Mock(return_value=["Line 1", "Line 2", "Line 3"])
        # self.remote_logs.clean = Mock(return_value=None)
        self.remote_logs.to_file("./logs/test_write_to_file.log")
        with open("./logs/test_write_to_file.log", "r", encoding="utf-8") as file:
            assert file.readlines()[:] == ["Line 1\n", "Line 2\n", "Line 3\n"]


if __name__ == "__main__":
    pytest.main()
