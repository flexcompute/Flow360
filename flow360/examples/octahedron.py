"""
octahdron geometry example
"""

from .base_test_case import BaseTestCase


class Octahedron(BaseTestCase):
    name = "octahedron"

    class url:
        geometry = "local://Trunc.SLDASM"
