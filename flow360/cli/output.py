"""Shared CLI output helpers."""

from __future__ import annotations

import json

import click


def emit_json(payload) -> None:
    """Emit stable pretty JSON for machine-readable commands."""

    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def emit_payload(payload, *, output_format: str = "json", text_formatter=None) -> None:
    """Emit a payload as JSON or through a presentation-only formatter."""

    if output_format == "json":
        emit_json(payload)
        return

    if output_format == "text" and text_formatter is not None:
        click.echo(text_formatter(payload))
        return

    raise click.ClickException(f"Unsupported output format: {output_format}")
