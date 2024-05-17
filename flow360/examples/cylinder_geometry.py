"""
cylinder meshing from geometry example
"""

from .base_test_case import BaseTestCase


class CylinderGeometry(BaseTestCase):
    name = "cylinder_geometry"

    class url:
        geometry = "local://cylinder.x_t"
        surface_json = "local://surface_params.json"
