from click.testing import CliRunner

from flow360.cli import flow360
import flow360.cli.logs as logs_module
from flow360.cloud.s3_utils import S3TransferType


class _FakeLogs:
    def __init__(self):
        self.calls = []

    def tail_lines(self, num_lines, chunk_size=None):
        self.calls.append(("tail", num_lines, chunk_size))
        return [f"tail:{num_lines}"]

    def head_lines(self, num_lines, chunk_size=None):
        self.calls.append(("head", num_lines, chunk_size))
        return [f"head:{num_lines}"]

    def read_all_text(self):
        self.calls.append(("all", None))
        return "all-text\n"


class _FakeResource:
    def __init__(self):
        self.logs = _FakeLogs()


def test_logs_defaults_to_tail(monkeypatch):
    resource = _FakeResource()
    monkeypatch.setattr(logs_module, "_resolve_logs_resource", lambda _: resource)
    runner = CliRunner()

    result = runner.invoke(flow360, ["logs", "case-1234567890abcdef"])

    assert result.exit_code == 0
    assert result.output == "tail:200\n"
    assert resource.logs.calls == [("tail", 200, 16384)]


def test_logs_head(monkeypatch):
    resource = _FakeResource()
    monkeypatch.setattr(logs_module, "_resolve_logs_resource", lambda _: resource)
    runner = CliRunner()

    result = runner.invoke(flow360, ["logs", "vm-1234567890abcdef", "--head", "5"])

    assert result.exit_code == 0
    assert result.output == "head:5\n"
    assert resource.logs.calls == [("head", 5, 1280)]


def test_logs_all(monkeypatch):
    resource = _FakeResource()
    monkeypatch.setattr(logs_module, "_resolve_logs_resource", lambda _: resource)
    runner = CliRunner()

    result = runner.invoke(flow360, ["logs", "sm-1234567890abcdef", "--all"])

    assert result.exit_code == 0
    assert result.output == "all-text\n"
    assert resource.logs.calls == [("all", None)]


def test_logs_save(monkeypatch, tmp_path):
    resource = _FakeResource()
    monkeypatch.setattr(logs_module, "_resolve_logs_resource", lambda _: resource)
    runner = CliRunner()
    output_path = tmp_path / "case.log"

    result = runner.invoke(
        flow360,
        ["logs", "case-1234567890abcdef", "--tail", "10", "--save", str(output_path)],
    )

    assert result.exit_code == 0
    assert result.output == f"Saved to {output_path}\n"
    assert output_path.read_text() == "tail:10\n"
    assert resource.logs.calls == [("tail", 10, 2560)]


def test_logs_rejects_multiple_modes():
    runner = CliRunner()

    result = runner.invoke(
        flow360,
        ["logs", "case-1234567890abcdef", "--tail", "10", "--head", "5"],
    )

    assert result.exit_code != 0
    assert "Use only one of --tail, --head, or --all." in result.output


def test_logs_rejects_unsupported_resource():
    runner = CliRunner()

    result = runner.invoke(flow360, ["logs", "prj-1234567890abcdef"])

    assert result.exit_code != 0
    assert "Unsupported resource id." in result.output


def test_resolve_logs_resource_uses_full_remote_log_paths():
    case_resource = logs_module._resolve_logs_resource("case-1234567890abcdef")
    volume_resource = logs_module._resolve_logs_resource("vm-1234567890abcdef")
    surface_resource = logs_module._resolve_logs_resource("sm-1234567890abcdef")

    assert case_resource.id == "case-1234567890abcdef"
    assert case_resource.s3_transfer_method is S3TransferType.CASE
    assert case_resource.logs._get_remote_log_file_name() == "logs/flow360_case.user.log"

    assert volume_resource.id == "vm-1234567890abcdef"
    assert volume_resource.s3_transfer_method is S3TransferType.VOLUME_MESH
    assert volume_resource.logs._get_remote_log_file_name() == "logs/flow360_volume_mesh.user.log"

    assert surface_resource.id == "sm-1234567890abcdef"
    assert surface_resource.s3_transfer_method is S3TransferType.SURFACE_MESH
    assert surface_resource.logs._get_remote_log_file_name() == "logs/flow360_surface_mesh.user.log"
