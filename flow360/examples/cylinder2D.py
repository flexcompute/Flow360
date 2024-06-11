"""
cylinder example
"""

from .base_test_case import BaseTestCase


class Cylinder(BaseTestCase):
    name = "cylinder"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/examples/cylinder/cylinder.cgns"
        mesh_json = "https://simcloud-public-1.s3.amazonaws.com/examples/cylinder/flow360mesh.json"
        case_json = "local://flow360.json"
        geometry = "local://cylinder.x_t"
        surface_json = "local://surface_params.json"
