"""Open Flow360 resources in the browser."""

from __future__ import annotations

import click

from flow360.cli.browser_links import get_resource_browser_payload, open_browser_url
from flow360.cli.output import emit_json
from flow360.cli.resource_refs import ResourceRefError


@click.command("open")
@click.argument("ref_id")
def open_resource(ref_id):
    """Open a Flow360 resource in the browser.

    Environment selection uses root options, for example: flow360 --dev open REF_ID.
    """
    try:
        payload = get_resource_browser_payload(ref_id)
    except ResourceRefError as error:
        raise click.ClickException(str(error)) from error

    payload["opened"] = open_browser_url(payload["url"])
    emit_json(payload)
