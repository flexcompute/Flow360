"""
airplane meshing example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class Airplane(BaseTestCase):
    name = "airplane"

    downloadable_assets = DownloadableAssets(
        geometry=Asset("local://geometry.csm"),
        surface_json=Asset("local://surface_params.json"),
        volume_json=Asset("local://volume_params.json"),
        case_json=Asset("local://case_params.json"),
    )
