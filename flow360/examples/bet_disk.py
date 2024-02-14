"""
betDisk example
"""

from .base_test_case import BaseTestCase


class BETDisk(BaseTestCase):
    name = "betDisk"

    class url:
        mesh = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/bodyBehindDisk.cgns"
        )
        mesh_json = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/flow360Mesh.json"
        )
        case_json = "local://flow360.json"
