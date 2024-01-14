class FeatureFlags:
    """
    Global feature flag config that allows import
    settings control before flow360 module is initialized.
    """

    def __init__(self):
        self._beta_feature = False

    def use_beta_features(self, value: bool):
        self._beta_feature = value

    def beta_features(self):
        return self._beta_feature


Flags = FeatureFlags()
