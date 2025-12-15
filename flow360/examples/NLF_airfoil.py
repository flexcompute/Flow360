"""
volume mesh NLF airfoil 2D example
"""

from .base_test_case import BaseTestCase


class NLFAirfoil2D(BaseTestCase):
    name = "nlf2d"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/examples/nlf/NLF_U1_2D.cgns"
