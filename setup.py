#!/usr/bin/env python3

import sys

import pkg_resources
import setuptools
import toml

"""

THIS IS HACKY setup.py FOR DEVELOPMENT ONLY. It is NOT recommended to use it and is not guaranteed to work.
For interactive installs, consider one of the alrenatives:
1. poetry install
2. pip install -e .

"""

if __name__ == "__main__":
    major, minor = sys.version_info.major, sys.version_info.minor
    with open("pyproject.toml", "r") as f:
        toml_data = toml.load(f)

    install_requires = []
    if "tool" in toml_data and "poetry" in toml_data["tool"]:
        parse_requires = toml_data["tool"]["poetry"].get("dependencies", {})
        for key, value in parse_requires.items():
            if key in ["python"]:
                continue
            if isinstance(value, list):
                version = [v["version"] for v in value if f"{major}.{minor}" in v["python"]]
                if len(version) == 0:
                    version = [v["version"] for v in value if f"{major}.{minor-1}" in v["python"]]
                version = version[0]
            else:
                version = value

            install_requires.append(f'{key}{version.replace("^", ">=")}')

        console_scripts = []
        parse_scripts = toml_data["tool"]["poetry"].get("scripts", {})
        for key, value in parse_scripts.items():
            console_scripts.append(f"{key} = {value}")

        entry_points = {"console_scripts": console_scripts}

        name = pkg_resources.safe_name(toml_data["tool"]["poetry"]["name"])
        version = toml_data["tool"]["poetry"]["version"]

    setuptools.setup(
        name=name,
        version=version,
        packages=setuptools.find_packages(exclude=["tests"]),
        install_requires=install_requires,
        entry_points=entry_points,
    )
