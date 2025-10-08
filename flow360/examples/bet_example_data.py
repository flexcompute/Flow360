"""
bet example
"""

from .base_test_case import BaseTestCase


class BETExampleData(BaseTestCase):
    name = "betExampleData"

    class url:
        mesh = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/bodyBehindDisk.cgns"
        )
        mesh_json = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/flow360Mesh.json"
        )
        extra = {"disk0": "disk0.json"}
