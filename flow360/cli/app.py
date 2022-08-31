"""
Commandline interface for flow360.
"""
import os.path
from os.path import expanduser

import click
import toml

home = expanduser("~")
if os.path.exists(f"{home}/.flow360/config.toml"):
    with open(f"{home}/.flow360/config.toml", "r", encoding="utf-8") as f:
        content = f.read()
        config = toml.loads(content)
        config_description = f"API Key[{config.get('default', {}).get('apikey', '')}]"


@click.group()
def flow360():
    """
    Commandline entrypoint for flow360.
    """


@click.command()
@click.option(
    "--apikey", prompt=config_description if "config_description" in globals() else "API Key"
)
def configure(apikey):
    """
    Configure flow360.
    :param apikey:
    :return:
    """

    with open(f"{home}/.flow360/config.toml", "w+", encoding="utf-8") as config_file:
        toml_config = toml.loads(config_file.read())
        toml_config.update({"default": {"apikey": apikey}})
        config_file.write(toml.dumps(toml_config))
        click.echo("done.")


flow360.add_command(configure)
