"""
bet example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class BETExampleData(BaseTestCase):
    name = "betExampleData"

    downloadable_assets = DownloadableAssets(
        mesh=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/bodyBehindDisk.cgns"
        ),
        mesh_json=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/flow360Mesh.json"
        ),
        extra={"disk0": "disk0.json"},
    )
