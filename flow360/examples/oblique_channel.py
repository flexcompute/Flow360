"""
Cartesian mesh with oblique boundaries example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class ObliqueChannel(BaseTestCase):
    name = "obliqueChannel"

    downloadable_assets = DownloadableAssets(
        mesh=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/obliqueChannel/cartesian_2d_mesh.oblique.cgns"
        ),
        extra={
            "rectangle_normal": "https://simcloud-public-1.s3.amazonaws.com/examples/obliqueChannel/rectangle_normal.cgns",
            "rectangle_oblique": "https://simcloud-public-1.s3.amazonaws.com/examples/obliqueChannel/rectangle_oblique.cgns",
        },
    )
