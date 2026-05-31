"""
cube geometry example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Cube(BaseTestCase):
    name = "cube"

    downloadable_assets = DownloadableAssets(
        geometry=Asset("https://simcloud-public-1.s3.amazonaws.com/cube/cube.stl"),
    )
