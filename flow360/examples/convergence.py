"""
convergence example
"""

from .base_test_case import BaseTestCase


class Convergence(BaseTestCase):
    name = "convergence"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/om6/wing_tetra.1.lb8.ugrid"
        mesh_json = "https://simcloud-public-1.s3.amazonaws.com/om6/Flow360Mesh.json"
        case_json = "local://flow360.json"
