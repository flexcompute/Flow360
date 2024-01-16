"""
Feature flags
"""

import os


class _FeatureFlags:
    def __init__(self):
        self._beta_features = os.environ.get("FLOW360_BETA_FEATURES", False)

    def beta_features(self):
        return self._beta_features


Flags = _FeatureFlags()
