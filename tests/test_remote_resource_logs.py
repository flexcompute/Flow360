import os
from unittest.mock import Mock

import pytest

from flow360.component.resource_base_v1 import (
    Flow360Resource,
    Position,
    RemoteResourceLogs,
)


def create_file(file_name: str, to_file):
    content = (
        ("[23-06-05 12:56:22][USER   ]:" + file_name + "\n") * 2
        + ("[23-06-05 23:24:43.075][USERDBG]:" + file_name + "\n") * 2
        + ("[23-06-05 12:55:46][USER   ]:WARNING:" + file_name + "\n") * 2
        + ("[23-06-05 12:55:48][USER   ]:CAPS Error:" + file_name + "\n") * 2
    ) * 3
    to_file = os.path.join(os.getcwd(), to_file)
    # encoded_content = content.encode("utf-8")  # Encode the string content to bytes
    with open(to_file, "w", encoding="utf-8") as file:
        file.write(content)


class TestRemoteResourceLogs:
    def setup_method(self):
        self.flow360_resource = Mock(spec=Flow360Resource)
        self.flow360_resource._download_file.side_effect = (
            lambda file_name, temp_file, overwrite: create_file(
                file_name=file_name, to_file=temp_file
            )
        )
        self.remote_logs = RemoteResourceLogs(self.flow360_resource)
        self.remote_logs.path = "file1.log"

    def test_get_log_by_pos(self):
        # Mock the necessary methods and attributes
        # Test head
        self.flow360_resource.get_download_file_list.return_value = [
            {"fileName": "logs/file1.log"},
        ]
        log_lines = self.remote_logs._get_log_by_pos(Position.HEAD, num_lines=5)
        assert log_lines == ["[23-06-05 12:56:22][USER   ]:file1.log"] * 2 + [
            "[23-06-05 23:24:43.075][USERDBG]:file1.log"
        ] * 2 + ["[23-06-05 12:55:46][USER   ]:WARNING:file1.log"]

        # Test tail with different number of lines
        # # Test all lines
        log_lines = self.remote_logs._get_log_by_pos()
        assert (
            log_lines
            == (
                ["[23-06-05 12:56:22][USER   ]:file1.log"] * 2
                + ["[23-06-05 23:24:43.075][USERDBG]:file1.log"] * 2
                + ["[23-06-05 12:55:46][USER   ]:WARNING:file1.log"] * 2
                + ["[23-06-05 12:55:48][USER   ]:CAPS Error:file1.log"] * 2
            )
            * 3
        )
        self.remote_logs.set_remote_log_file_name("file2.log")
        log_lines = self.remote_logs._get_log_by_pos(Position.TAIL, num_lines=2)
        assert log_lines == ["[23-06-05 12:55:48][USER   ]:CAPS Error:file2.log"] * 2

    def test_get_log_by_level(self):
        # Mock the necessary methods and attributes
        self.flow360_resource.get_download_file_list.return_value = [{"fileName": "logs/file1.log"}]

        # Test error
        log_lines = self.remote_logs._get_log_by_level("ERROR")
        assert log_lines == ["[23-06-05 12:55:48][USER   ]:CAPS Error:file1.log"] * 6

        log_lines = self.remote_logs._get_log_by_level("WARNING")
        assert (
            log_lines
            == (
                ["[23-06-05 12:55:46][USER   ]:WARNING:file1.log"] * 2
                + ["[23-06-05 12:55:48][USER   ]:CAPS Error:file1.log"] * 2
            )
            * 3
        )

        log_lines = self.remote_logs._get_log_by_level("INFO")
        assert (
            log_lines
            == (
                ["[23-06-05 12:56:22][USER   ]:file1.log"] * 2
                + ["[23-06-05 12:55:46][USER   ]:WARNING:file1.log"] * 2
                + ["[23-06-05 12:55:48][USER   ]:CAPS Error:file1.log"] * 2
            )
            * 3
        )
        log_lines = self.remote_logs._get_log_by_level()
        assert (
            log_lines
            == (
                ["[23-06-05 12:56:22][USER   ]:file1.log"] * 2
                + ["[23-06-05 23:24:43.075][USERDBG]:file1.log"] * 2
                + ["[23-06-05 12:55:46][USER   ]:WARNING:file1.log"] * 2
                + ["[23-06-05 12:55:48][USER   ]:CAPS Error:file1.log"] * 2
            )
            * 3
        )
        self.remote_logs.set_remote_log_file_name("file2.log")
        log_lines = self.remote_logs._get_log_by_level("INFO")
        assert (
            log_lines
            == (
                ["[23-06-05 12:56:22][USER   ]:file2.log"] * 2
                + ["[23-06-05 12:55:46][USER   ]:WARNING:file2.log"] * 2
                + ["[23-06-05 12:55:48][USER   ]:CAPS Error:file2.log"] * 2
            )
            * 3
        )

    def test_head_tail_print(self):
        # Mock the necessary methods and attributes
        self.remote_logs._get_log_by_pos = Mock(return_value=["Line 1", "Line 2", "Line 3"])
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
        # Call the method under test
        self.remote_logs.errors()
        self.remote_logs._get_log_by_level.assert_called_with("ERROR")
        self.remote_logs.warnings()
        self.remote_logs._get_log_by_level.assert_called_with("WARNING")
        self.remote_logs.info()
        self.remote_logs._get_log_by_level.assert_called_with("INFO")

    def test_write_to_file(self):
        self.flow360_resource.get_download_file_list.return_value = [
            {"fileName": "logs/file1.log"},
        ]
        temp_file = os.path.join(os.getcwd(), "test_write_to_file_temp_file")

        self.remote_logs.to_file(temp_file)

        with open(temp_file) as temp:
            with open(self.remote_logs._get_tmp_file_name()) as original_file:
                temp.seek(0)
                assert temp.read() == original_file.read()
        os.remove(temp_file)


if __name__ == "__main__":
    pytest.main()
