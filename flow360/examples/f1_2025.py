"""
F1 2025 example
"""

from .base_test_case import BaseTestCase


class F1_2025(BaseTestCase):
    name = "f1_2025"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025/f1_2025_m.stl.zst"
