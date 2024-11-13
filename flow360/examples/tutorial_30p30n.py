"""
Tutorial_30p30n meshing example
"""

from .base_test_case import BaseTestCase


class Tutorial_30p30n(BaseTestCase):
    name = "tutorial_30p30n"

    class url:
        geometry = "local://geometry.csm"
        surface_json = "local://surface_params.json"
        volume_json = "local://volume_params.json"
        case_json = "local://case_params.json"
