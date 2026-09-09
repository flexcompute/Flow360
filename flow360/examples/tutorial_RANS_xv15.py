"""
tutorial_calculating dynamic derivatives using sliding interfaces example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TutorialRANSXv15(BaseTestCase):
    name = "tutorialRANSXv15"

    downloadable_assets = DownloadableAssets(
        mesh=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/xv15/XV15_Hover_ascent_coarse_v2.cgns"
        ),
    )
