"""
isolated propeller with a nacelle example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class IsolatedPropellerNacelle(BaseTestCase):
    name = "isolatedPropellerNacelle"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/isolated_propeller/isolated_propeller_nacelle.csm"
        ),
    )
