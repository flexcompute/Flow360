"""
quadcopter example
"""

from .base_test_case import BaseTestCase


class Quadcopter(BaseTestCase):
    name = "quadcopter"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/quadcopter/quadcopter.cgns.zst"
