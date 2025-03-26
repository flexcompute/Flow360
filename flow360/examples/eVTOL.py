"""
eVTOL quickstart example
"""

from .base_test_case import BaseTestCase


class eVTOL(BaseTestCase):
    name = "eVTOL"

    class url:
        geometry = "local://evtol_quickstart_grouped.csm"
