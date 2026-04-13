"""
Commandline interface for flow360.
"""

import os
from datetime import datetime

import click
import toml
from packaging.version import InvalidVersion, Version

from flow360.cli.auth import LoginError, resolve_target_environment, wait_for_login
from flow360.environment import Env
from flow360.user_config import (
    config_file,
    delete_apikey,
    read_user_config,
    store_apikey,
    write_user_config,
)
from flow360.version import __solver_version__, __version__

if os.path.exists(config_file):
    with open(config_file, encoding="utf-8") as current_fh:
        current_config = toml.loads(current_fh.read())
        saved_apikey = current_config.get("default", {}).get("apikey", None)
        if saved_apikey is not None:
            APIKEY_PRESENT = True


@click.group()
def flow360():
    """
    Commandline entrypoint for flow360.
    """


@click.command("configure", context_settings={"show_default": True})
@click.option(
    "--apikey", prompt=False if "APIKEY_PRESENT" in globals() else "API Key", help="API Key"
)
@click.option("--profile", prompt=False, default="default", help="Profile, e.g., default, secondary.")
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
def configure(apikey, profile, dev, uat, env, suppress_submit_warning, beta_features):
    """
    Configure flow360.
    """
    changed = False
    config = read_user_config()
    _, storage_environment = resolve_target_environment(dev=dev, uat=uat, env=env)

    if apikey is not None:
        config = store_apikey(apikey, profile=profile, environment_name=storage_environment)
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
@click.option("--profile", prompt=False, default="default", help="Profile, e.g., default, secondary.")
@click.option("--dev", prompt=False, type=bool, is_flag=True, help="Log in to DEV.")
@click.option("--uat", prompt=False, type=bool, is_flag=True, help="Log in to UAT.")
@click.option(
    "--local",
    prompt=False,
    type=bool,
    is_flag=True,
    hidden=True,
    help="Open the local DEV frontend at local.dev-simulation.cloud:3000 and store the key under DEV.",
)
@click.option("--env", prompt=False, default=None, help="Log in to a named environment.")
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=None,
    help="Fixed localhost callback port. Defaults to an ephemeral port.",
)
@click.option("--timeout", type=click.IntRange(1, 3600), default=120, help="Login timeout in seconds.")
def login(profile, dev, uat, local, env, port, timeout):
    """
    Open a browser login flow and store the resulting API key.
    """
    def announce_login(details):
        click.echo(f"Starting local login server on {details['callback_url']}.")
        if details["browser_opened"] == "true":
            click.echo("If your browser did not open, navigate to this URL to authenticate:")
        else:
            click.echo("Could not open your browser automatically. Navigate to this URL to authenticate:")
        click.echo("")
        click.echo(details["login_url"])
        click.echo("")

    try:
        environment, _ = resolve_target_environment(dev=dev, uat=uat, env=env, local=local)
        result = wait_for_login(
            environment=environment,
            profile=profile,
            port=port,
            timeout=timeout,
            use_local_ui=local,
            announce_login=announce_login,
        )
    except (LoginError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if result.get("email"):
        click.echo(f"Successfully logged in as {result['email']}")
    else:
        click.echo("Successfully logged in")


@click.command("logout", context_settings={"show_default": True})
@click.option("--profile", prompt=False, default="default", help="Profile, e.g., default, secondary.")
@click.option("--dev", prompt=False, type=bool, is_flag=True, help="Remove the DEV login.")
@click.option("--uat", prompt=False, type=bool, is_flag=True, help="Remove the UAT login.")
@click.option(
    "--local",
    prompt=False,
    type=bool,
    is_flag=True,
    hidden=True,
    help="Remove the local DEV login (same stored target as DEV).",
)
@click.option("--env", prompt=False, default=None, help="Remove the login for a named environment.")
def logout(profile, dev, uat, local, env):
    """
    Remove a stored Flow360 API key.
    """
    try:
        environment, storage_environment = resolve_target_environment(dev=dev, uat=uat, env=env, local=local)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    removed, _ = delete_apikey(profile=profile, environment_name=storage_environment)
    if not removed:
        click.echo(
            f"No stored API key found for profile '{profile}' in environment '{environment.name}'."
        )
        return

    click.echo(
        f"Removed stored API key for profile '{profile}' in environment '{environment.name}'."
    )


# For displaying all projects
@click.command("show_projects", context_settings={"show_default": True})
@click.option("--keyword", "-k", help="Filter projects by keyword", default=None, type=str)
@click.option("--env", prompt=False, default=None, help="The environment used for the query.")
def show_projects(keyword, env: str):
    """
    Display all available projects with optional filter.
    """
    prev_env_config = None
    if env:
        env_config = Env.load(env)
        prev_env_config = Env.current
        env_config.active()

    # pylint: disable=import-outside-toplevel
    from flow360.component.simulation.web.project_records import (
        show_projects_with_keyword_filter,
    )

    show_projects_with_keyword_filter(search_keyword=keyword)

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
