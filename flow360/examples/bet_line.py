"""
betLine example
"""


from .base_test_case import BaseTestCase


class BETLine(BaseTestCase):
    name = "betLine"

    class url:
        mesh = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/bodyBehindDisk.cgns"
        )
        mesh_json = (
            "https://simcloud-public-1.s3.amazonaws.com/examples/actuatorDisk/flow360Mesh.json"
        )
        case_json = "local://flow360.json"
