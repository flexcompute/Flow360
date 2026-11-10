"""
cylinder2D example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Cylinder2D(BaseTestCase):
    name = "cylinder2D"

    downloadable_assets = DownloadableAssets(
        mesh=Asset("https://simcloud-public-1.s3.amazonaws.com/examples/cylinder/cylinder.cgns"),
        mesh_json=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/cylinder/flow360mesh.json"
        ),
        case_json=Asset("local://flow360.json"),
    )
