"""
cube geometry example
"""

from .base_test_case import BaseTestCase


class cube(BaseTestCase):
    name = "cube"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/cube/cube.stl"
