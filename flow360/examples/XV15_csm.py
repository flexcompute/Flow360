"""
XV-15 csm file geometry
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class XV15_CSM(BaseTestCase):
    name = "XV15_CSM"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/xv15_wing/XV15_Wing.csm"
        ),
    )
