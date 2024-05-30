"""
cylinder2D example
"""

from .base_test_case import BaseTestCase


class Cylinder2D(BaseTestCase):
    name = "cylinder2D"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/examples/cylinder/cylinder.cgns"
        mesh_json = "https://simcloud-public-1.s3.amazonaws.com/examples/cylinder/flow360mesh.json"
        case_json = "local://flow360.json"
