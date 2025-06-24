"""
isolated propeller example
"""

from .base_test_case import BaseTestCase


class IsolatedPropeller(BaseTestCase):
    name = "isolatedPropeller"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/examples/isolated_propeller/isolated_propeller.csm"
