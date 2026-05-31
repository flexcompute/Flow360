"""
Windsor geometry
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Windsor(BaseTestCase):
    name = "Windsor"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            [
                "https://simcloud-public-1.s3.amazonaws.com/windsor/windsorBody.stp",
                "https://simcloud-public-1.s3.amazonaws.com/windsor/wheel.stl",
            ]
        )
    )
