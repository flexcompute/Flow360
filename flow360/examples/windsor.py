"""
Windsor geometry
"""

from .base_test_case import BaseTestCase


class Windsor(BaseTestCase):
    name = "Windsor"

    class url:
        wheel = "https://simcloud-public-1.s3.amazonaws.com/windsor/wheel.stl"
        body = "https://simcloud-public-1.s3.amazonaws.com/windsor/windsorBody.stp"