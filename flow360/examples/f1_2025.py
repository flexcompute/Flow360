"""
F1 2025 example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class F1_2025(BaseTestCase):
    name = "f1_2025"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(
            [
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/body.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/eb.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/fr-int.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/fr-susp.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/fr-wh.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/fw.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/inflows_outflows.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/rr-int.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/rr-susp.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/rr-wh.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/rw.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/toint-rad.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/tunnel.stl.zst",
                "https://simcloud-public-1.s3.amazonaws.com/examples/f1_2025_split/uf.stl.zst",
            ]
        ),
    )
