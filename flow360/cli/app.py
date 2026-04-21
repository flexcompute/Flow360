"""
Commandline interface for flow360.
"""

from __future__ import annotations

import os
from datetime import datetime
from importlib import import_module

import click
import toml
from packaging.version import InvalidVersion, Version

# Importing through ``flow360`` eagerly loads the SDK public surface.
# pylint: disable=consider-using-from-import
import flow360.user_config as user_config
from flow360.cli.context import merge_command_context, resolve_root_context
from flow360.environment import Env
from flow360.user_config import (
    delete_apikey,
    read_user_config,
    store_apikey,
    write_user_config,
)
from flow360.version import __solver_version__, __version__

config_file = user_config.config_file

_LAZY_COMMANDS = {
    "project": {
        "module": "flow360.cli.project",
        "attr": "project",
        "help": "Inspect and manage Flow360 projects.",
    },
    "draft": {
        "module": "flow360.cli.draft",
        "attr": "draft",
        "help": "Inspect draft resources.",
    },
    "geometry": {
        "module": "flow360.cli.assets",
        "attr": "geometry",
        "help": "Inspect and manage Flow360 geometries.",
    },
    "surface-mesh": {
        "module": "flow360.cli.assets",
        "attr": "surface_mesh",
        "help": "Inspect and manage Flow360 surface meshes.",
    },
    "volume-mesh": {
        "module": "flow360.cli.assets",
        "attr": "volume_mesh",
        "help": "Inspect and manage Flow360 volume meshes.",
    },
    "case": {
        "module": "flow360.cli.assets",
        "attr": "case",
        "help": "Inspect and manage Flow360 cases.",
    },
    "folder": {
        "module": "flow360.cli.folder",
        "attr": "folder",
        "help": "Inspect Flow360 folders.",
    },
    "open": {
        "module": "flow360.cli.open_resource",
        "attr": "open_resource",
        "help": "Open a Flow360 resource in the browser.",
    },
    "wait": {
        "module": "flow360.cli.wait",
        "attr": "wait",
        "help": "Wait for a Flow360 resource to reach a terminal state.",
    },
}


class LazyFlow360Group(click.Group):
    """Click group that imports SDK-backed command groups on demand."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except click.ClickException:
            raise
        except Exception as error:  # pylint: disable=broad-except
            # Convert uncaught SDK auth failures into normal CLI errors.
            # pylint: disable=import-outside-toplevel
            from flow360.exceptions import Flow360AuthorisationError

            if isinstance(error, Flow360AuthorisationError):
                raise click.ClickException(str(error)) from error
            raise

    def list_commands(self, ctx):
        return sorted(set(super().list_commands(ctx)) | set(_LAZY_COMMANDS))

    def format_commands(self, ctx, formatter):
        rows = []
        for command_name in self.list_commands(ctx):
            command = self.commands.get(command_name)
            if command is not None:
                help_text = command.get_short_help_str(formatter.width)
            else:
                help_text = _LAZY_COMMANDS[command_name]["help"]
            rows.append((command_name, help_text))

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def get_command(self, ctx, cmd_name):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        loader = _LAZY_COMMANDS.get(cmd_name)
        if loader is None:
            return None

        module_name = loader["module"]
        attr_name = loader["attr"]
        command = getattr(import_module(module_name), attr_name)
        self.add_command(command, cmd_name)
        return command


def _has_stored_apikey(config: dict, profile: str, environment_name: str | None) -> bool:
    profile_config = config.get(profile) or {}
    if environment_name is None:
        return bool(profile_config.get("apikey"))
    return bool((profile_config.get(environment_name) or {}).get("apikey"))


@click.group(cls=LazyFlow360Group)
@click.option("--profile", default=None, help="API key profile for requests.")
@click.option("--dev", is_flag=True, help="Use the DEV environment.")
@click.option("--uat", is_flag=True, help="Use the UAT environment.")
@click.option("--env", default=None, help="Use a named environment.")
@click.pass_context
def flow360(ctx, profile, dev, uat, env):
    """
    Commandline entrypoint for flow360.
    """
    ctx.ensure_object(dict)
    resolved_context = resolve_root_context(profile=profile, dev=dev, uat=uat, env=env)

    prev_env = Env.current
    prev_profile = user_config.UserConfig.profile
    prev_profile_env = os.environ.get("SIMCLOUD_PROFILE")

    if profile is not None:
        os.environ["SIMCLOUD_PROFILE"] = profile
        user_config.UserConfig.set_profile(profile)

        def restore_profile():
            if prev_profile_env is None:
                os.environ.pop("SIMCLOUD_PROFILE", None)
            else:
                os.environ["SIMCLOUD_PROFILE"] = prev_profile_env
            user_config.UserConfig.set_profile(prev_profile)

        ctx.call_on_close(restore_profile)

    if dev or uat or env is not None:
        # pylint: disable=import-outside-toplevel
        from flow360.cli.auth import resolve_target_environment

        try:
            environment, _ = resolve_target_environment(dev=dev, uat=uat, env=env)
        except ValueError as error:
            raise click.ClickException(str(error)) from error

        environment.active()

        def restore_env():
            prev_env.active()

        ctx.call_on_close(restore_env)

    ctx.obj.update(resolved_context.as_dict())


@click.command("configure", context_settings={"show_default": True})
@click.pass_context
@click.option("--apikey", prompt=False, help="API Key")
@click.option("--profile", prompt=False, default=None, help="Profile, e.g., default, secondary.")
@click.option(
    "--dev", prompt=False, type=bool, is_flag=True, help="Only use this apikey in DEV environment."
)
@click.option(
    "--uat", prompt=False, type=bool, is_flag=True, help="Only use this apikey in UAT environment."
)
@click.option("--env", prompt=False, default=None, help="Only use this apikey in this environment.")
@click.option(
    "--suppress-submit-warning",
    type=bool,
    help='Whether to suppress warnings for "submit()" when creating new Case, new VolumeMesh etc.',
)
@click.option(
    "--beta-features",
    type=bool,
    help="Toggle beta features support",
)
# pylint: disable=too-many-arguments, too-many-branches
def configure(ctx, apikey, profile, dev, uat, env, suppress_submit_warning, beta_features):
    """
    Configure flow360.
    """
    changed = False
    config = read_user_config()
    cli_context = merge_command_context(ctx, profile=profile, dev=dev, uat=uat, env=env)

    # pylint: disable=import-outside-toplevel
    from flow360.cli.auth import resolve_target_environment

    _, storage_environment = resolve_target_environment(
        dev=cli_context.dev, uat=cli_context.uat, env=cli_context.env
    )

    if (
        apikey is None
        and suppress_submit_warning is None
        and beta_features is None
        and not _has_stored_apikey(config, cli_context.profile, storage_environment)
    ):
        apikey = click.prompt("API Key")

    if apikey is not None:
        config = store_apikey(
            apikey,
            profile=cli_context.profile,
            environment_name=storage_environment,
        )
        changed = True

    if suppress_submit_warning is not None:
        config.setdefault("user", {}).setdefault("config", {})[
            "suppress_submit_warning"
        ] = suppress_submit_warning
        changed = True

    if beta_features is not None:
        config.setdefault("user", {}).setdefault("config", {})["beta_features"] = beta_features
        changed = True

    write_user_config(config)

    if not changed:
        click.echo("Nothing to do. Your current config:")
        click.echo(toml.dumps(config))
        click.echo("run flow360 configure --help to see options")
    click.echo("done.")


@click.command("login", context_settings={"show_default": True})
@click.pass_context
@click.option("--profile", prompt=False, default=None, help="Profile, e.g., default, secondary.")
@click.option("--dev", prompt=False, type=bool, is_flag=True, help="Log in to DEV.")
@click.option("--uat", prompt=False, type=bool, is_flag=True, help="Log in to UAT.")
@click.option("--env", prompt=False, default=None, help="Log in to a named environment.")
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=None,
    help="Fixed localhost callback port. Defaults to an ephemeral port.",
)
@click.option(
    "--timeout", type=click.IntRange(1, 3600), default=120, help="Login timeout in seconds."
)
def login(
    ctx, profile, dev, uat, env, port, timeout
):  # pylint: disable=too-many-arguments,too-many-locals
    """
    Open a browser login flow and store the resulting API key.
    """
    # pylint: disable=import-outside-toplevel
    from flow360.cli.auth import LoginError, resolve_target_environment, wait_for_login
    from flow360.cli.auth_guidance import build_configure_command

    def announce_login(details):
        click.echo(f"Starting local login server on {details['callback_url']}.")
        if details["browser_opened"] == "true":
            click.echo("If your browser did not open, navigate to this URL to authenticate:")
        else:
            click.echo(
                "Could not open your browser automatically. Navigate to this URL to authenticate:"
            )
        click.echo("")
        click.echo(details["login_url"])
        click.echo("")
        click.echo("Headless environment? Configure an API key manually with:")
        click.echo(f"  {build_configure_command(details['environment'], details['profile'])}")
        click.echo("")

    try:
        cli_context = merge_command_context(
            ctx,
            profile=profile,
            dev=dev,
            uat=uat,
            env=env,
        )
        environment, _ = resolve_target_environment(
            dev=cli_context.dev,
            uat=cli_context.uat,
            env=cli_context.env,
        )
        result = wait_for_login(
            environment=environment,
            profile=cli_context.profile,
            port=port,
            timeout=timeout,
            announce_login=announce_login,
        )
    except (LoginError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if result.get("email"):
        click.echo(f"Successfully logged in as {result['email']}")
    else:
        click.echo("Successfully logged in")


@click.command("logout", context_settings={"show_default": True})
@click.pass_context
@click.option("--profile", prompt=False, default=None, help="Profile, e.g., default, secondary.")
@click.option("--dev", prompt=False, type=bool, is_flag=True, help="Remove the DEV login.")
@click.option("--uat", prompt=False, type=bool, is_flag=True, help="Remove the UAT login.")
@click.option("--env", prompt=False, default=None, help="Remove the login for a named environment.")
def logout(ctx, profile, dev, uat, env):  # pylint: disable=too-many-arguments
    """
    Remove a stored Flow360 API key.
    """
    # pylint: disable=import-outside-toplevel
    from flow360.cli.auth import resolve_target_environment

    try:
        cli_context = merge_command_context(
            ctx,
            profile=profile,
            dev=dev,
            uat=uat,
            env=env,
        )
        environment, storage_environment = resolve_target_environment(
            dev=cli_context.dev,
            uat=cli_context.uat,
            env=cli_context.env,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    removed, _ = delete_apikey(profile=cli_context.profile, environment_name=storage_environment)
    if not removed:
        click.echo(
            f"No stored API key found for profile '{cli_context.profile}' in environment '{environment.name}'."
        )
        return

    click.echo(
        f"Removed stored API key for profile '{cli_context.profile}' in environment '{environment.name}'."
    )


# For displaying all projects
@click.command("show_projects", context_settings={"show_default": True})
@click.pass_context
@click.option("--keyword", "-k", help="Filter projects by keyword", default=None, type=str)
@click.option("--env", prompt=False, default=None, help="The environment used for the query.")
def show_projects(ctx, keyword, env: str):
    """
    Display all available projects with optional filter.
    """
    prev_env_config = None
    cli_context = merge_command_context(ctx, env=env)
    if cli_context.env:
        env_config = Env.load(cli_context.env)
        prev_env_config = Env.current
        env_config.active()

    # pylint: disable=import-outside-toplevel
    from flow360.cli.project import (
        build_project_list_payload,
        format_project_list_payload,
    )

    payload = build_project_list_payload(
        search=keyword,
        limit=200,
        folder_ids=(),
        exclude_subfolders=False,
    )
    click.echo(format_project_list_payload(payload))

    if prev_env_config:
        prev_env_config.active()


@click.command("version")
def version():  # pylint: disable=too-many-locals, too-many-statements
    """
    Display the version of the flow360 client,
    plus available versions for each solver release.
    """
    # Fetch PyPI data
    # pylint: disable=import-outside-toplevel
    from collections import defaultdict

    from requests import RequestException, get

    try:
        resp = get("https://pypi.org/pypi/flow360/json", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except RequestException:
        click.echo("Failed to fetch PyPI data.")
        click.echo(f"Installed version: {__version__}")
        click.echo(f"Default solver version: {__solver_version__}")
        return

    # Parse versions
    parsed = []
    for version_string in data["releases"].keys():
        try:
            parsed.append(Version(version_string))
        except InvalidVersion:
            continue

    # Group by solver version (major.minor)
    versions_by_solver = defaultdict(list)
    for v in parsed:
        solver_key = f"release-{v.major}.{v.minor}"
        versions_by_solver[solver_key].append(v)

    sorted_solver_keys = sorted(
        versions_by_solver.keys(),
        key=lambda s: [int(x) for x in s.replace("release-", "").split(".")],
        reverse=True,
    )

    def get_release_date(ver: Version) -> str:
        releases = data["releases"].get(str(ver), [])
        times = [
            fi.get("upload_time_iso_8601") or fi.get("upload_time")
            for fi in releases
            if fi.get("upload_time_iso_8601") or fi.get("upload_time")
        ]
        if not times:
            return "-"
        dates = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]
        d = min(dates).date()
        return f"{d.strftime('%b')} {d.day:>2}, {d.year}"

    # Prepare table
    headers = ("Solver Version", "Installed", "Latest Stable", "Released", "Latest Beta (Unstable)")
    rows = []

    blacklist = ["release-25.4"]

    for solver_ver in sorted_solver_keys:
        # Internal filter: only show 24.11+
        parts = [int(x) for x in solver_ver.replace("release-", "").split(".")]
        if parts < [24, 11]:
            continue

        if solver_ver in blacklist:
            continue

        versions = versions_by_solver[solver_ver]

        stables = [v for v in versions if not v.is_prerelease]
        betas = [v for v in versions if v.is_prerelease]

        latest_stable = max(stables) if stables else None
        latest_beta = max(betas) if betas else None

        # If beta is older than stable, don't show it
        if latest_stable and latest_beta and latest_beta < latest_stable:
            latest_beta = None

        # Only show release date for stable versions
        release_date = get_release_date(latest_stable) if latest_stable else "-"

        installed_str = str(__version__) if solver_ver == __solver_version__ else "-"
        stable_str = str(latest_stable) if latest_stable else "-"
        beta_str = str(latest_beta) if latest_beta else "-"

        rows.append((solver_ver, installed_str, stable_str, release_date, beta_str))

    # Compute column widths
    w1 = max(len(r[0]) for r in rows + [headers])
    w2 = max(len(r[1]) for r in rows + [headers])
    w3 = max(len(r[2]) for r in rows + [headers])
    w4 = max(len(r[3]) for r in rows + [headers])
    w5 = max(len(r[4]) for r in rows + [headers])

    # Print header
    header_line = (
        f"{headers[0].ljust(w1)}  {headers[1].ljust(w2)}  {headers[2].ljust(w3)}  "
        f"{headers[3].ljust(w4)}  {headers[4].ljust(w5)}"
    )
    click.echo(header_line)
    click.echo(f"{'-'*w1}  {'-'*w2}  {'-'*w3}  {'-'*w4}  {'-'*w5}")

    # Print data rows
    for sv, inst, stable, date, beta in rows:
        line = f"{sv.ljust(w1)}  {inst.ljust(w2)}  {stable.ljust(w3)}  {date.ljust(w4)}  {beta.ljust(w5)}"
        if sv == __solver_version__:
            click.echo(click.style(line, fg="green", bold=True))
        else:
            click.echo(line)


flow360.add_command(configure)
flow360.add_command(login)
flow360.add_command(logout)
flow360.add_command(show_projects)
flow360.add_command(version)
