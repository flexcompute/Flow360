"""
Flow360 Module
This module provides utilities for working with the Flow360 application.
Module Attributes:
    home (str): The user's home directory.
    flow360_dir (str): The path to the Flow360 directory.
"""

import os

home = os.path.expanduser("~")
# pylint: disable=invalid-name
flow360_dir = f"{home}/.flow360/"
