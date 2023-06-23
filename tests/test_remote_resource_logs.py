import os
import unittest
from unittest.mock import Mock
from flow360.component.resource_base import RemoteResourceLogs, Flow360Resource, Position
from flow360.log import LogLevel
import unittest
from unittest import mock


class TestRemoteResourceLogs(unittest.TestCase):
    def create_file(self, fnames):
        os.makedirs("./logs/", exist_ok=True)
        for fname in fnames:
            with open("./logs/" + fname, "w") as file:
                for _ in range(3):
                    file.write(("Info:" + fname + "\n") * 2)
                    file.write(("Debug:" + fname + "\n") * 2)
                    file.write(("Warning:" + fname + "\n") * 2)
                    file.write(("Error:" + fname + "\n") * 2)

    def setUp(self):
        self.flow360 = Mock(spec=Flow360Resource)
        self.remote_logs = RemoteResourceLogs(self.flow360)

    def set_up_files(self):
        self.remote_logs.download_log = Mock(
            return_value=self.create_file(["file1.log", "file2.log"])
        )
        self.remote_logs.dir_path = "./logs/"
        self.remote_logs.path = "./logs/file1.log"
        self.remote_logs.paths = ["./logs/file1.log"]

    def test_download_log(self):
        # Mock the necessary methods and attributes
        self.flow360.get_download_file_list = Mock(return_value=["file1.log", "file2.log"])
        self.flow360.download_file.return_value = self.create_file(["file1.log", "file2.log"])

        # Call the method under test
        self.remote_logs.download_log(to_file="./logs/")

        # Assertions
        self.assertEqual(self.remote_logs.dir_path, "./logs/")
        self.assertEqual(self.remote_logs.paths, ["./logs/file1.log", "./logs/file2.log"])
        self.remote_logs.dir_path = "./logs/"
        self.remote_logs.clean()

    def test_clean(self):
        # Mock the necessary methods and attributes
        self.create_file(["file.log"])
        self.remote_logs.dir_path = "./logs/"
        self.remote_logs.path = "./logs/file.log"

        # Call the method under test
        self.remote_logs.clean()

        # Assertions
        assert not os.path.exists("./logs")
        assert not os.path.exists("./logs/file.log")
        self.assertIsNone(self.remote_logs.path)

    def test_get_log_by_pos(self):
        # Mock the necessary methods and attributes
        self.set_up_files()
        # Test head
        log_lines = self.remote_logs.get_log_by_pos(Position.HEAD, num_lines=5)
        self.assertEqual(
            log_lines,
            ["Info:file1.log\n"] * 2 + ["Debug:file1.log\n"] * 2 + ["Warning:file1.log\n"],
        )

        # Test tail with different number of lines
        self.remote_logs.paths = ["./logs/file1.log", "./logs/file2.log"]
        log_lines = self.remote_logs.get_log_by_pos(
            Position.TAIL, num_lines=2, fname="./logs/file2.log"
        )
        self.assertEqual(
            log_lines,
            ["Error:file2.log\n"] * 2,
        )

        # Test all lines
        log_lines = self.remote_logs.get_log_by_pos(fname="./logs/file1.log")
        self.assertEqual(
            log_lines,
            (
                ["Info:file1.log\n"] * 2
                + ["Debug:file1.log\n"] * 2
                + ["Warning:file1.log\n"] * 2
                + ["Error:file1.log\n"] * 2
            )
            * 3,
        )
        self.remote_logs.clean()

    def test_get_log_by_level(self):
        # Mock the necessary methods and attributes
        self.set_up_files()
        self.remote_logs.path = "./logs/file1.log"

        # Test error
        log_lines = self.remote_logs.get_log_by_level("ERROR")
        self.assertEqual(log_lines, ["Error:file1.log"] * 6)

        log_lines = self.remote_logs.get_log_by_level("WARNING")
        self.assertEqual(log_lines, (["Warning:file1.log"] * 2 + ["Error:file1.log"] * 2) * 3)

        self.remote_logs.paths = ["./logs/file1.log", "./logs/file2.log"]
        log_lines = self.remote_logs.get_log_by_level("INFO", "./logs/file2.log")
        self.assertEqual(
            log_lines,
            (["Info:file2.log"] * 2 + ["Warning:file2.log"] * 2 + ["Error:file2.log"] * 2) * 3,
        )

        log_lines = self.remote_logs.get_log_by_level(fname="./logs/file1.log")
        self.assertEqual(
            log_lines,
            (
                ["Info:file1.log"] * 2
                + ["Debug:file1.log"] * 2
                + ["Warning:file1.log"] * 2
                + ["Error:file1.log"] * 2
            )
            * 3,
        )
        self.remote_logs.clean()

    def test_head_tail_print(self):
        # Mock the necessary methods and attributes
        self.remote_logs.get_log_by_pos = Mock(return_value=["Line 1", "Line 2", "Line 3"])
        self.remote_logs.clean = Mock(return_value=None)
        # Head
        self.remote_logs.head(num_lines=2)
        self.remote_logs.get_log_by_pos.assert_called_with(Position.HEAD, 2)
        # Tail
        self.remote_logs.tail(num_lines=2)
        self.remote_logs.get_log_by_pos.assert_called_with(Position.TAIL, 2)
        # Print
        self.remote_logs.print()
        self.remote_logs.get_log_by_pos.assert_called_with(Position.ALL)

    def test_errors_warnings_info(self):
        # Mock the necessary methods and attributes
        self.remote_logs.get_log_by_level = Mock(
            return_value=[
                "Error: Something went wrong",
                "Warning: This is a warning",
                "Info: Hey",
            ]
        )
        self.remote_logs.clean = Mock(return_value=None)
        # Call the method under test
        self.remote_logs.errors()
        self.remote_logs.get_log_by_level.assert_called_with("ERROR")
        self.remote_logs.warnings()
        self.remote_logs.get_log_by_level.assert_called_with("WARNING")
        self.remote_logs.info()
        self.remote_logs.get_log_by_level.assert_called_with("INFO")

    def test_write_to_file(self):
        os.makedirs("./logs/", exist_ok=True)
        self.remote_logs.get_log_by_pos = Mock(return_value=["Line 1", "Line 2", "Line 3"])
        self.remote_logs.clean = Mock(return_value=None)
        self.remote_logs.log_to_file("./logs/test_write_to_file.log")
        with open("./logs/test_write_to_file.log", "r", encoding="utf-8") as file:
            assert file.readlines()[:] == ["Line 1\n", "Line 2\n", "Line 3\n"]
