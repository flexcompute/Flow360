"""
evtol example
"""

from .base_test_case import BaseTestCase


class EVTOL(BaseTestCase):
    name = "evtol"

    class url:
        geometry = (
            "https://simcloud-public-1.s3.amazonaws.com/quickstart/evtol_quickstart_grouped.csm"
        )
        mesh = "https://simcloud-public-1.s3.amazonaws.com/quickstart/evtol_quickstart.cgns.zst"
