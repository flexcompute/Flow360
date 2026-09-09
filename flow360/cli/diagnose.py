"""
Connection diagnostics CLI command.
"""

from __future__ import annotations

import click

from flow360.cli.output import emit_json


@click.command("diagnose")
@click.option(
    "--env",
    "env_name",
    default=None,
    help="Diagnose a named environment (dev/uat/prod/preprod or one saved in config.toml).",
)
@click.option(
    "--profile",
    default=None,
    help="API key profile to diagnose with, e.g., default, secondary.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON instead of text.")
@click.pass_context
def diagnose(ctx, env_name, profile, as_json):
    """Verify connectivity and authentication to a Flow360 deployment layer by layer.

    Diagnoses a fully configured environment: define and save it first (see the
    on-premises setup guide), store its API key with `flow360 configure`, then
    run this command.
    """
    # pylint: disable=import-outside-toplevel
    from flow360.cli.context import activate_profile
    from flow360.diagnostics import diagnose as run_diagnose
    from flow360.environment import Env

    if profile is not None:
        activate_profile(ctx, profile)

    if env_name:
        try:
            environment = Env.load(env_name)
        except (ValueError, FileNotFoundError) as error:
            raise click.ClickException(str(error)) from error
        source = "--env (config.toml)"
    else:
        environment = Env.current
        source = "the active environment"

    report = run_diagnose(environment, source=source, print_report=not as_json)
    if as_json:
        emit_json(report.model_dump(mode="json"))

    if not report.ok:
        ctx.exit(1)
