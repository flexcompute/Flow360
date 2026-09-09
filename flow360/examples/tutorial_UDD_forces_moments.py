"""
tutorial of UDD forces and moments example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TutorialUDDForcesMoments(BaseTestCase):
    name = "tutorialUDDForcesMoments"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/tutorials/user_defined_postprocessing/UDD_FM_airplane.csm"
        ),
    )
