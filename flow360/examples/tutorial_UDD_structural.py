"""
tutorial of UDD forces and moments example
"""

from .base_test_case import BaseTestCase


class TutorialUDDStructural(BaseTestCase):
    name = "tutorialUDDStructural"

    class url:
        geometry = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/rotatingPlate/rotatingPlate.cgns"
        )
