from unittest.mock import Mock

from click.testing import CliRunner

from flow360.cli import flow360
from flow360.component.resource_base import Flow360Resource, RemoteResourceLogs


class _FakeLogs:
    def __init__(self):
        self.calls = []
        self.tail_output = ["line 2", "line 3"]

    def read_all_text(self):
        self.calls.append(("all",))
        return "line 1\nline 2\n"

    def head_lines(self, num_lines, chunk_size):
        self.calls.append(("head", num_lines, chunk_size))
        return ["line 1", "line 2"][:num_lines]

    def tail_lines(self, num_lines, chunk_size):
        self.calls.append(("tail", num_lines, chunk_size))
        return self.tail_output[-num_lines:]


class _FakeLogsResource:
    def __init__(self):
        self.logs = _FakeLogs()


def _read_text_range(content: bytes, byte_range):
    if byte_range is None:
        body = content
        start_index = 0
    else:
        start, end = byte_range
        if start is not None and start < 0 and end is None:
            start_index = max(len(content) + start, 0)
            body = content[start_index:]
        elif end is None:
            start_index = start
            body = content[start:]
        else:
            start_index = start
            body = content[start : end + 1]
    end_index = start_index + len(body) - 1
    return body.decode("utf-8", errors="replace"), {
        "body_length": len(body),
        "content_range": f"bytes {start_index}-{end_index}/{len(content)}",
    }


def _remote_resource_logs_for_content(content: bytes):
    flow360_resource = Mock(spec=Flow360Resource)
    flow360_resource.id = "case-123"
    flow360_resource.s3_transfer_method = Mock()
    flow360_resource.s3_transfer_method.read_text.side_effect = (
        lambda resource_id, remote_file_name, byte_range=None: _read_text_range(content, byte_range)
    )
    logs = RemoteResourceLogs(flow360_resource)
    logs.set_remote_log_file_name("file1.log")
    return logs


def test_logs_default_tails_case_logs(monkeypatch):
    from flow360.cli import logs as logs_cli

    runner = CliRunner()
    resource = _FakeLogsResource()
    monkeypatch.setattr(logs_cli, "_resolve_logs_resource", lambda resource_id: resource)

    result = runner.invoke(flow360, ["logs", "case-123"])

    assert result.exit_code == 0
    assert result.output == "line 2\nline 3\n"
    assert resource.logs.calls == [("tail", 200, 16 * 1024)]


def test_logs_preserves_trailing_empty_lines(monkeypatch):
    from flow360.cli import logs as logs_cli

    runner = CliRunner()
    resource = _FakeLogsResource()
    resource.logs.tail_output = ["line 1", ""]
    monkeypatch.setattr(logs_cli, "_resolve_logs_resource", lambda resource_id: resource)

    result = runner.invoke(flow360, ["logs", "case-123", "--tail", "2"])

    assert result.exit_code == 0
    assert result.output == "line 1\n\n"


def test_logs_supports_head_tail_all_and_save(monkeypatch, tmp_path):
    from flow360.cli import logs as logs_cli

    runner = CliRunner()
    resource = _FakeLogsResource()
    output_file = tmp_path / "case.log"
    monkeypatch.setattr(logs_cli, "_resolve_logs_resource", lambda resource_id: resource)

    head_result = runner.invoke(flow360, ["logs", "case-123", "--head", "1"])
    all_result = runner.invoke(flow360, ["logs", "case-123", "--all", "--save", str(output_file)])

    assert head_result.exit_code == 0
    assert head_result.output == "line 1\n"
    assert all_result.exit_code == 0
    assert output_file.read_text(encoding="utf-8") == "line 1\nline 2\n"
    assert resource.logs.calls == [("head", 1, 1024), ("all",)]


def test_logs_rejects_multiple_modes():
    runner = CliRunner()

    result = runner.invoke(flow360, ["logs", "case-123", "--head", "1", "--tail", "1"])

    assert result.exit_code != 0
    assert "Use only one" in result.output


def test_logs_rejects_unsupported_resource():
    runner = CliRunner()

    result = runner.invoke(flow360, ["logs", "geo-123"])

    assert result.exit_code != 0
    assert "Unsupported resource id" in result.output


def test_range_line_reads_use_byte_lengths_for_non_ascii_logs():
    logs = _remote_resource_logs_for_content("é\nline 2\nline 3\n".encode("utf-8"))

    assert logs.head_lines(2, chunk_size=4) == ["é", "line 2"]
    assert logs.tail_lines(2, chunk_size=4) == ["line 2", "line 3"]


def test_range_line_reads_do_not_retry_at_exact_probe_boundary():
    calls = []
    logs = _remote_resource_logs_for_content(b"abcde")
    logs.flow360_resource.s3_transfer_method.read_text.side_effect = (
        lambda resource_id, remote_file_name, byte_range=None: (
            calls.append(byte_range) or _read_text_range(b"abcde", byte_range)
        )
    )

    assert logs.head_lines(1, chunk_size=4) == ["abcde"]
    assert calls == [(0, 4)]

    calls.clear()
    assert logs.tail_lines(1, chunk_size=4) == ["abcde"]
    assert calls == [(-5, None)]


def test_range_line_reads_stop_on_empty_or_non_growing_reads():
    logs = _remote_resource_logs_for_content(b"")
    logs.flow360_resource.s3_transfer_method.read_text.side_effect = None
    logs.flow360_resource.s3_transfer_method.read_text.return_value = (
        "",
        {"body_length": 0, "content_range": "bytes 0-0/10"},
    )

    assert logs.head_lines(1, chunk_size=4) == []
    assert logs.tail_lines(1, chunk_size=4) == []

    calls = []
    logs.flow360_resource.s3_transfer_method.read_text.side_effect = (
        lambda resource_id, remote_file_name, byte_range=None: (
            calls.append(byte_range)
            or ("abcd", {"body_length": 4, "content_range": "bytes 0-3/10"})
        )
    )

    assert logs.head_lines(1, chunk_size=4) == []
    assert calls == [(0, 4), (0, 8)]

    calls.clear()
    assert logs.tail_lines(1, chunk_size=4) == []
    assert calls == [(-5, None), (-9, None)]
