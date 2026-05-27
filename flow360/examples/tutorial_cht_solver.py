"""
tutorialPeriodicBC example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TutorialCHTSolver(BaseTestCase):
    name = "tutorialCHTSolver"

    downloadable_assets = DownloadableAssets(
        mesh=Asset("https://simcloud-public-1.s3.amazonaws.com/tutorials/cht/cooling-fins_v2.cgns"),
    )
