"""
Cartesian mesh with oblique boundaries example
"""

from .base_test_case import BaseTestCase


class ObliqueChannel(BaseTestCase):
    name = "obliqueChannel"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/examples/obliqueChannel/cartesian_2d_mesh.oblique.cgns"
        extra = {
            "rectangle_normal": "https://simcloud-public-1.s3.amazonaws.com/examples/obliqueChannel/rectangle_normal.cgns",
            "rectangle_oblique": "https://simcloud-public-1.s3.amazonaws.com/examples/obliqueChannel/rectangle_oblique.cgns",
        }
