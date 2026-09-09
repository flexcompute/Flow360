"""
rotatingSpheres example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class RotatingSpheres(BaseTestCase):
    name = "rotatingSpheres"

    downloadable_assets = DownloadableAssets(
        mesh=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/rotatingSpheres/spheres.cgns"
        ),
        mesh_json=Asset("local://flow360mesh.json"),
        case_json=Asset("local://flow360.json"),
    )
