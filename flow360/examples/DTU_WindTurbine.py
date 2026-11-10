"""
DTU Wind Turbine example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class DTU_WindTurbine(BaseTestCase):
    name = "dtu_wind_turbine"

    downloadable_assets = DownloadableAssets(
        geometry=Asset("https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/geometry.egads"),
    )
