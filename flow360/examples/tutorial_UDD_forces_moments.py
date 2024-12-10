"""
tutorial of UDD forces and moments example
"""

from .base_test_case import BaseTestCase


class TutorialUDDForcesMoments(BaseTestCase):
    name = "tutorialUDDForcesMoments"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/tutorials/user_defined_postprocessing/UDD_FM_airplane.csm"
