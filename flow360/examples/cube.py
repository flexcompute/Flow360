"""
cube geometry example
"""

from .base_test_case import BaseTestCase


class Cube(BaseTestCase):
    name = "cube"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/cube/cube.stl"
