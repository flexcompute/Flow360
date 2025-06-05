"""
tutorialBetDisk meshing example
"""

from .base_test_case import BaseTestCase


class TutorialBETDisk(BaseTestCase):
    name = "tutorialBetDisk"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/betTutorial/BET_tutorial_wing.csm"
        extra = {"xrotor": "xv15.xrotor"}
