"""
tutorial2DCrm meshing example
"""

from .base_test_case import BaseTestCase


class Tutorial2DCRM(BaseTestCase):
    name = "tutorial2DCrm"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/tutorials/2d_multielement/2D_CRM_geometry.csm"
