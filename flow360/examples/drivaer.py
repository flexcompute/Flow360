"""
drivaer example
"""

from .base_test_case import BaseTestCase


class DrivAer(BaseTestCase):
    name = "drivaer"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/drivaer/drivaer.cgns.zst"
