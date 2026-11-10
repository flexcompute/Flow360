"""
Generic resource wait command.
"""

from __future__ import annotations

import click

from flow360.cli import resource_state
from flow360.cli.output import emit_json


@click.command("wait")
@click.argument("ref_id")
@click.option(
    "--timeout",
    default=3600,
    show_default=True,
    type=click.FloatRange(min=0.1, min_open=False),
    help="Maximum wait time in seconds.",
)
@click.option(
    "--poll-interval",
    default=2.0,
    show_default=True,
    type=click.FloatRange(min=0.1, min_open=False),
    help="Polling interval in seconds.",
)
def wait(ref_id, timeout, poll_interval):
    """Wait for a resource to reach a terminal state."""
    try:
        state = resource_state.wait_for_resource_state(
            ref_id, timeout=timeout, poll_interval=poll_interval
        )
    except resource_state.WaitTimeoutError as error:
        payload = dict(error.state or {})
        payload["timed_out"] = True
        emit_json(payload)
        raise click.exceptions.Exit(124) from error

    emit_json(state)
    if not state["is_success"]:
        raise click.exceptions.Exit(1)
