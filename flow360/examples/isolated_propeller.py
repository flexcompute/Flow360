"""
isolated propeller example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class IsolatedPropeller(BaseTestCase):
    name = "isolatedPropeller"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/isolated_propeller/isolated_propeller.csm"
        ),
    )
