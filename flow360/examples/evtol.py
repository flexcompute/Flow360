"""
evtol example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class EVTOL(BaseTestCase):
    name = "evtol"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/quickstart/evtol_quickstart_grouped.csm"
        ),
        mesh=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/quickstart/evtol_quickstart.cgns.zst"
        ),
    )
