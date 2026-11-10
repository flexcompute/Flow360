"""
quadcopter example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Quadcopter(BaseTestCase):
    name = "quadcopter"

    downloadable_assets = DownloadableAssets(
        mesh=Asset("https://simcloud-public-1.s3.amazonaws.com/quadcopter/quadcopter.cgns.zst"),
    )
