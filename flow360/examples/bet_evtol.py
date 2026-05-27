"""
bet evtol example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class BETEVTOL(BaseTestCase):
    name = "betEVTOL"

    downloadable_assets = DownloadableAssets(
        geometry=Asset("https://simcloud-public-1.s3.amazonaws.com/examples/bet_evtol/evtol.egads"),
        extra={
            "disk13": "disk13.json",
            "disk24": "disk24.json",
            "disk57": "disk57.json",
            "disk68": "disk68.json",
        },
    )
