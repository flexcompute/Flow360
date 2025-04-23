"""
Commandline interface for flow360.
"""

import os.path
from os.path import expanduser

import click
import toml

from flow360.cli import dict_utils
from flow360.component.project_utils import show_projects_with_keyword_filter
from flow360.environment import Env

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
    Display all available projects with optional keyword filter.
    """
    prev_env_config = None
    if env:
        env_config = Env.load(env)
        prev_env_config = Env.current
        env_config.active()

    show_projects_with_keyword_filter(search_keyword=keyword)

    if prev_env_config:
        prev_env_config.active()


flow360.add_command(configure)
flow360.add_command(show_projects)
