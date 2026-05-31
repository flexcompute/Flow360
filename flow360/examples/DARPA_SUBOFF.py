"""
DARPA SUBOFF geometry
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class DARPA_SUBOFF(BaseTestCase):
    name = "DARPA_SUBOFF"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/submarine/DARPA_SUBOFF.STEP"
        ),
    )
