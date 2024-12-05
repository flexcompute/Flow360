"""
tutorial of UDD alpha controller example
"""

from .base_test_case import BaseTestCase


class TutorialUDDAlphaController(BaseTestCase):
    name = "tutorialUDDAlphaController"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/om6/wing_tetra.1.lb8.ugrid"
