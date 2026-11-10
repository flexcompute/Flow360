"""
Feature flags
"""

import ast
import os

import toml

from .file_path import flow360_dir

config_file = os.path.join(flow360_dir, "config.toml")


# pylint: disable=too-few-public-methods
class _FeatureFlags:
    def __init__(self):
        self.config = {}
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as file_handler:
                self.config = toml.loads(file_handler.read())

        self._beta_features = os.environ.get("FLOW360_BETA_FEATURES", None)

        if self._beta_features is None:
            self._beta_features = (
                self.config.get("user", {}).get("config", {}).get("beta_features", False)
            )
        else:
            self._beta_features = ast.literal_eval(self._beta_features)

    def beta_features(self):
        """Does flow360 support beta features?"""
        return self._beta_features


Flags = _FeatureFlags()
