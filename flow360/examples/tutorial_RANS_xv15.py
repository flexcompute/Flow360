"""
tutorial_calculating dynamic derivatives using sliding interfaces example
"""

from .base_test_case import BaseTestCase


class TutorialRANSXv15(BaseTestCase):
    name = "tutorialRANSXv15"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/xv15/XV15_Hover_ascent_coarse_v2.cgns"
