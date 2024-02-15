"""
Commandline interface for flow360.
"""

import os.path
from os.path import expanduser

import click
import toml

from flow360.cli import dict_utils

home = expanduser("~")
config_file = f"{home}/.flow360/config.toml"

if os.path.exists(config_file):
    with open(config_file, "r", encoding="utf-8") as current_fh:
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
    "--dev", prompt=False, type=bool, is_flag=True, help="Only use this apikey in dev environment."
)
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
def configure(apikey, profile, dev, suppress_submit_warning, beta_features):
    """
    Configure flow360.
    """
    changed = False
    if not os.path.exists(f"{home}/.flow360"):
        os.makedirs(f"{home}/.flow360")

    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as file_handler:
            config = toml.loads(file_handler.read())

    if apikey is not None:
        entry = {profile: {"apikey": apikey}} if not dev else {profile: {"dev": {"apikey": apikey}}}
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


flow360.add_command(configure)
