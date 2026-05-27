"""
om6wing example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class OM6wing(BaseTestCase):
    name = "om6wing"

    downloadable_assets = DownloadableAssets(
        mesh=Asset("https://simcloud-public-1.s3.amazonaws.com/om6/wing_tetra.1.lb8.ugrid"),
        mesh_json=Asset("https://simcloud-public-1.s3.amazonaws.com/om6/Flow360Mesh.json"),
        case_json=Asset("local://flow360.json"),
        case_yaml=Asset("local://case.yaml"),
    )
