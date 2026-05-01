"""Open Flow360 resources in the browser."""

from __future__ import annotations

import click

from flow360.cli.browser_links import get_resource_browser_payload, open_browser_url
from flow360.cli.output import emit_json
from flow360.cli.resource_refs import ResourceRefError


@click.command("open")
@click.argument("ref_id")
@click.option(
    "--workspace-id",
    default=None,
    hidden=True,
    help="Internal override for folder workspace resolution.",
)
def open_resource(ref_id, workspace_id):
    """Open a Flow360 resource in the browser."""
    try:
        payload = get_resource_browser_payload(ref_id, workspace_id=workspace_id)
    except ResourceRefError as error:
        raise click.ClickException(str(error)) from error

    payload["opened"] = open_browser_url(payload["url"])
    emit_json(payload)
