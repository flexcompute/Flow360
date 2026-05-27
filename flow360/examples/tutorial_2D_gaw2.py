"""
tutorial2DGaw2 meshing example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Tutorial2DGAW2(BaseTestCase):
    name = "tutorial2DGaw2"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/tutorials/2d_multielement/2D_GAW2_geometry.csm"
        ),
    )
