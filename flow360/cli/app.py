"""
Commandline interface for flow360.
"""

import os.path
from datetime import datetime
from os.path import expanduser

import click
import toml
from packaging.version import InvalidVersion, Version

from flow360.cli import dict_utils
from flow360.component.project_utils import show_projects_with_keyword_filter
from flow360.environment import Env
from flow360.version import __solver_version__, __version__

home = expanduser("~")
# pylint: disable=invalid-name
config_file = f"{home}/.flow360/config.toml"

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
@click.option("--profile", prompt=False, default="default", help="Profile, e.g., default, dev.")
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
    if not os.path.exists(f"{home}/.flow360"):
        os.makedirs(f"{home}/.flow360")

    config = {}
    if os.path.exists(config_file):
        with open(config_file, encoding="utf-8") as file_handler:
            config = toml.loads(file_handler.read())

    if apikey is not None:
        if dev is True:
            entry = {profile: {"dev": {"apikey": apikey}}}
        elif uat is True:
            entry = {profile: {"uat": {"apikey": apikey}}}
        elif env:
            if env == "dev":
                raise ValueError("Cannot set dev environment with --env, please use --dev instead.")
            if env == "uat":
                raise ValueError("Cannot set uat environment with --env, please use --uat instead.")
            if env == "prod":
                raise ValueError(
                    "Cannot set prod environment with --env, please remove --env and its argument."
                )
            entry = {profile: {env: {"apikey": apikey}}}
        else:
            entry = {profile: {"apikey": apikey}}
        dict_utils.merge_overwrite(config, entry)
        changed = True

    if suppress_submit_warning is not None:
        dict_utils.merge_overwrite(
            config, {"user": {"config": {"suppress_submit_warning": suppress_submit_warning}}}
        )
        changed = True

    if beta_features is not None:
        dict_utils.merge_overwrite(config, {"user": {"config": {"beta_features": beta_features}}})
        changed = True

    with open(config_file, "w", encoding="utf-8") as file_handler:
        file_handler.write(toml.dumps(config))

    if not changed:
        click.echo("Nothing to do. Your current config:")
        click.echo(toml.dumps(config))
        click.echo("run flow360 configure --help to see options")
    click.echo("done.")


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
flow360.add_command(show_projects)
flow360.add_command(version)
