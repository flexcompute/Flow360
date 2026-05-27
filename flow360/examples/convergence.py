"""
convergence example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Convergence(BaseTestCase):
    name = "convergence"

    downloadable_assets = DownloadableAssets(
        mesh=Asset("https://simcloud-public-1.s3.amazonaws.com/om6/wing_tetra.1.lb8.ugrid"),
        mesh_json=Asset("https://simcloud-public-1.s3.amazonaws.com/om6/Flow360Mesh.json"),
        case_json=Asset("local://flow360.json"),
    )
