"""
tutorialUDDDynamicGridRotation meshing example
"""

from .base_test_case import BaseTestCase


class TutorialUDDDynamicGridRotation(BaseTestCase):
    name = "tutorialUDDDynamicGridRotation"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/udd/rotatingPlate.cgns"