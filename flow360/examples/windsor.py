"""
Windsor geometry
"""

from .base_test_case import BaseTestCase


class Windsor(BaseTestCase):
    name = "Windsor"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/windsor/windsorBody.stp"
        extra = {"wheel": "https://simcloud-public-1.s3.amazonaws.com/windsor/wheel.stl"}
