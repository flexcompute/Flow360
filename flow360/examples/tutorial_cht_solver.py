"""
tutorialPeriodicBC example
"""

from .base_test_case import BaseTestCase


class TutorialCHTSolver(BaseTestCase):
    name = "tutorialCHTSolver"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/tutorials/cht/cooling-fins_v2.cgns"
