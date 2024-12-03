"""
tutorialPeriodicBC example
"""

from .base_test_case import BaseTestCase


class TutorialPeriodicBC(BaseTestCase):
    name = "tutorialPeriodicBC"

    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/tutorials/periodic_boundary_condition/periodic_boundary_condition_tu_berlin_stator_mesh.cgns.gz"
