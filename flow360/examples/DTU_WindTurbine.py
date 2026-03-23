"""
DTU Wind Turbine example
"""

from .base_test_case import BaseTestCase


class DTU_WindTurbine(BaseTestCase):
    name = "dtu_wind_turbine"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/geometry.egads"
