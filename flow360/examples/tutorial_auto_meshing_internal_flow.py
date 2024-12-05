"""
tutorial_automated meshing for internal flow example
"""

from .base_test_case import BaseTestCase


class TutorialAutoMeshingInternalFlow(BaseTestCase):
    name = "tutorialAutoMeshingInternalFlow"

    class url:
        geometry = "https://simcloud-public-1.s3.amazonaws.com/tutorials/auto_meshing_internal_flow/internalFlow.csm"
