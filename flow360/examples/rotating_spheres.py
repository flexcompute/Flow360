"""
rotatingSpheres example
"""

from .base_test_case import BaseTestCase


class RotatingSpheres(BaseTestCase):
    name = "rotatingSpheres"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/examples/rotatingSpheres/spheres.cgns"
        mesh_json = "local://flow360mesh.json"
        case_json = "local://flow360.json"
