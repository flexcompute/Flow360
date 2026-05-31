"""
cylinder3D example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Cylinder3D(BaseTestCase):
    name = "cylinder3D"

    downloadable_assets = DownloadableAssets(
        geometry=Asset("local://cylinder.x_t"),
        surface_json=Asset("local://surface_params.json"),
        case_json=Asset("local://case_params.json"),
    )
