"""
Shared CLI command factory for Flow360 resources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class ResourceCommandSpec:
    """Specification for common resource inspection commands."""

    command_name: str
    id_argument: str
    help_text: str
    label: str
    emit_info: Callable[[str], None]
    emit_state: Callable[[str], None]
    emit_simulation_params: Callable[[str], None]
    normalize_id: Callable[[str], str] = lambda resource_id: resource_id
    emit_summary: Callable[[str], None] | None = None


def make_resource_group(spec: ResourceCommandSpec):
    """Create a Click group with common resource read commands."""

    @click.group(spec.command_name, help=spec.help_text)
    def resource_group():
        pass

    def get_resource_id(kwargs):
        return spec.normalize_id(kwargs[spec.id_argument])

    @resource_group.command("info", help=f"Get {spec.label} metadata.")
    @click.argument(spec.id_argument)
    def info_command(**kwargs):
        spec.emit_info(get_resource_id(kwargs))

    @resource_group.command("state", help=f"Get {spec.label} lifecycle state.")
    @click.argument(spec.id_argument)
    def state_command(**kwargs):
        spec.emit_state(get_resource_id(kwargs))

    if spec.emit_summary is not None:

        @resource_group.command(
            "summary",
            help=f"Summarize validated {spec.label} SimulationParams user intent.",
        )
        @click.argument(spec.id_argument)
        def summary_command(**kwargs):
            spec.emit_summary(get_resource_id(kwargs))

    @resource_group.group("simulation-params", help="Namespace for SimulationParams commands.")
    def simulation_params_group():
        pass

    @simulation_params_group.command(
        "get",
        help=f"Get raw {spec.label} SimulationParams JSON.",
    )
    @click.argument(spec.id_argument)
    def simulation_params_get_command(**kwargs):
        spec.emit_simulation_params(get_resource_id(kwargs))

    return resource_group
