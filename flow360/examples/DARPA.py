"""
DARPA SUBOFF geometry
"""

from .base_test_case import BaseTestCase


class DARPA_SUBOFF(BaseTestCase):
    name = "DARPA_SUBOFF"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/examples/submarine/DARPA.STEP"
