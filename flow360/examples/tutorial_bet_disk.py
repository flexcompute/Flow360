"""
tutorialBetDisk meshing example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TutorialBETDisk(BaseTestCase):
    name = "tutorialBetDisk"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/betTutorial/BET_tutorial_wing.csm"
        ),
        extra={"xrotor": "xv15.xrotor"},
    )
