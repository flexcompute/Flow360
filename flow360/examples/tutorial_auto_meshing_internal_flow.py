"""
tutorial_automated meshing for internal flow example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TutorialAutoMeshingInternalFlow(BaseTestCase):
    name = "tutorialAutoMeshingInternalFlow"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/tutorials/auto_meshing_internal_flow/internalFlow.csm"
        ),
    )
