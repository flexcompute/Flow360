"""
CLI command for fetching logs from Flow360 resources.
"""

from __future__ import annotations

from pathlib import Path

import click


class _LogsResource:  # pylint: disable=too-few-public-methods
    def __init__(self, resource_id, s3_transfer_method, remote_log_file_name):
        # pylint: disable=import-outside-toplevel
        from flow360.component.resource_base import RemoteResourceLogs

        self.id = resource_id
        self.s3_transfer_method = s3_transfer_method
        self.logs = RemoteResourceLogs(self)
        self.logs.set_remote_log_file_name(remote_log_file_name)


def _normalize_output(text: str) -> str:
    if text and not text.endswith("\n"):
        return text + "\n"
    return text


def _render_lines(lines) -> str:
    return "\n".join(lines) + "\n" if lines else ""


def _preferred_chunk_size(num_lines: int) -> int:
    return max(1024, min(16 * 1024, num_lines * 256))


def _resolve_logs_resource(resource_id: str):
    # pylint: disable=import-outside-toplevel
    from flow360.cloud.s3_utils import S3TransferType

    if resource_id.startswith("case-"):
        return _LogsResource(resource_id, S3TransferType.CASE, "logs/flow360_case.user.log")

    if resource_id.startswith("vm-"):
        return _LogsResource(
            resource_id,
            S3TransferType.VOLUME_MESH,
            "logs/flow360_volume_mesh.user.log",
        )

    if resource_id.startswith("sm-"):
        return _LogsResource(
            resource_id,
            S3TransferType.SURFACE_MESH,
            "logs/flow360_surface_mesh.user.log",
        )

    raise click.ClickException(
        "Unsupported resource id. Logs currently support Case (case-...), "
        "VolumeMesh (vm-...), and SurfaceMesh (sm-...) resources."
    )


@click.command("logs", context_settings={"show_default": True})
@click.argument("resource_id")
@click.option("--tail", "tail_lines", type=click.IntRange(1, None), help="Show the last N lines.")
@click.option("--head", "head_lines", type=click.IntRange(1, None), help="Show the first N lines.")
@click.option("--all", "show_all", is_flag=True, help="Show the entire log.")
@click.option(
    "--save",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the selected output to a file instead of stdout.",
)
def logs(resource_id, tail_lines, head_lines, show_all, save):
    """
    Fetch logs for a completed or running resource.
    """
    selected_modes = sum(
        mode is not None and mode is not False for mode in (tail_lines, head_lines, show_all)
    )
    if selected_modes > 1:
        raise click.ClickException("Use only one of --tail, --head, or --all.")

    if tail_lines is None and head_lines is None and not show_all:
        tail_lines = 200

    resource = _resolve_logs_resource(resource_id)
    if show_all:
        output = _normalize_output(resource.logs.read_all_text())
    elif head_lines is not None:
        output = _render_lines(
            resource.logs.head_lines(head_lines, chunk_size=_preferred_chunk_size(head_lines))
        )
    else:
        output = _render_lines(
            resource.logs.tail_lines(tail_lines, chunk_size=_preferred_chunk_size(tail_lines))
        )

    if save is not None:
        save.write_text(output, encoding="utf-8")
        click.echo(f"Saved to {save}")
        return

    click.echo(output, nl=False)
