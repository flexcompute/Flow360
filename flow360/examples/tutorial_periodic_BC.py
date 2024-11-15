"""
tutorialPeriodicBC example
"""

from .base_test_case import BaseTestCase


class TutorialPeriodicBC(BaseTestCase):
    name = "tutorialPeriodicBC"

    class url:
        mesh = "local://volume_mesh.cgns.gz"
