"""
Commandline interface for flow360.
"""
import os.path
from os.path import expanduser

import click
import toml

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
    "--suppress-submit-warning",
    type=bool,
    is_flag=True,
    help='Whether to suppress warnings for "submit()" when creating new Case, new VolumeMesh etc.',
)
@click.option(
    "--show-submit-warning",
    type=bool,
    is_flag=True,
    help='Whether to show warnings for "submit()" when creating new Case, new VolumeMesh etc.',
)
def configure(apikey, profile, suppress_submit_warning, show_submit_warning):
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
        config.update({profile: {"apikey": apikey}})
        changed = True

    if suppress_submit_warning and show_submit_warning:
        raise click.ClickException(
            "You cannot use both --suppress-submit-warning AND --show-submit-warning"
        )

    if suppress_submit_warning:
        config.update({"user": {"config": {"suppress_submit_warning": True}}})
        changed = True

    if show_submit_warning:
        config.update({"user": {"config": {"suppress_submit_warning": False}}})
        changed = True

    with open(config_file, "w", encoding="utf-8") as file_handler:
        file_handler.write(toml.dumps(config))

    if not changed:
        click.echo("Nothing to do. Your current config:")
        click.echo(toml.dumps(config))
        click.echo("run flow360 configure --help to see options")
    click.echo("done.")


flow360.add_command(configure)
