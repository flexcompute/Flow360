"""
tutorial2D30p30n meshing example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Tutorial2D30p30n(BaseTestCase):
    name = "tutorial2D30p30n"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/tutorials/2d_multielement/2D_30p30n_geometry.csm"
        ),
    )
