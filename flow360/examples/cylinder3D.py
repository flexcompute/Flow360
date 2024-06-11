"""
cylinder3D example
"""

from .base_test_case import BaseTestCase


class Cylinder3D(BaseTestCase):
    name = "cylinder3D"

    class url:
        geometry = "local://cylinder.x_t"
        surface_json = "local://surface_params.json"
        case_json = "local://case_params.json"
