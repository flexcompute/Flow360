import os
import requests

here = os.path.dirname(os.path.abspath(__file__))


def download(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as fh:
        fh.write(response.content)


class OM6test:
    class url:
        mesh = "https://simcloud-public-1.s3.amazonaws.com/om6/wing_tetra.1.lb8.ugrid"
        mesh_json = "https://simcloud-public-1.s3.amazonaws.com/om6/Flow360Mesh.json"
        case_json = "https://simcloud-public-1.s3.amazonaws.com/om6/Flow360.json"

    _mesh_filename = os.path.join(here, os.path.basename(url.mesh))
    _mesh_json = os.path.join(here, os.path.basename(url.mesh_json))
    _case_json = os.path.join(here, os.path.basename(url.case_json))

    @classmethod
    def is_file_downloaded(cls, file):
        if not os.path.exists(file):
            raise RuntimeError("File not found. Run OM6test.get_files() first to download files.")
        return file

    @classmethod
    @property
    def mesh_filename(cls):
        return cls.is_file_downloaded(cls._mesh_filename)

    @classmethod
    @property
    def mesh_json(cls):
        return cls.is_file_downloaded(cls._mesh_json)

    @classmethod
    @property
    def case_json(cls):
        return cls.is_file_downloaded(cls._case_json)

    @classmethod
    def get_files(cls):
        if not os.path.exists(cls._mesh_filename):
            download(cls.url.mesh, cls._mesh_filename)
        if not os.path.exists(cls._mesh_json):
            download(cls.url.mesh_json, cls._mesh_json)
        if not os.path.exists(cls._case_json):
            download(cls.url.case_json, cls._case_json)
