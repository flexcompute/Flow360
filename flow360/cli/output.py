"""Shared CLI output helpers."""

from __future__ import annotations

import json

import click


def emit_json(payload) -> None:
    """Emit stable pretty JSON for machine-readable commands."""

    click.echo(json.dumps(payload, indent=2, sort_keys=True))
