"""
actuatorDisk example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class ActuatorDisk(BaseTestCase):
    name = "actuatorDisk"

    downloadable_assets = DownloadableAssets(
        mesh=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/bodyBehindDisk.cgns"
        ),
        mesh_json=Asset(
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/flow360Mesh.json"
        ),
        case_json=Asset("local://flow360.json"),
    )
