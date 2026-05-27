"""
drivaer example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class DrivAer(BaseTestCase):
    name = "drivaer"

    downloadable_assets = DownloadableAssets(
        mesh=Asset("https://simcloud-public-1.s3.amazonaws.com/drivaer/drivaer.cgns.zst"),
    )
