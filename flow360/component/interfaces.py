"""webAPI interface definitions
"""

from typing import Union

from pydantic.v1 import BaseModel

from ..cloud.s3_utils import S3TransferType


class BaseInterface(BaseModel):
    """
    Basic interface for endpoint definition.
    """

    resource_type: str
    s3_transfer_method: Union[S3TransferType, None]
    endpoint: str


SurfaceMeshInterface = BaseInterface(
    resource_type="Surface Mesh",
    s3_transfer_method=S3TransferType.SURFACE_MESH,
    endpoint="surfacemeshes",
)


VolumeMeshInterface = BaseInterface(
    resource_type="Volume Mesh",
    s3_transfer_method=S3TransferType.VOLUME_MESH,
    endpoint="volumemeshes",
)

CaseInterface = BaseInterface(
    resource_type="Case", s3_transfer_method=S3TransferType.CASE, endpoint="cases"
)

GeometryInterface = BaseInterface(
    resource_type="Geometry",
    s3_transfer_method=S3TransferType.GEOMETRY,
    endpoint="v2/geometries",
)

ProjectInterface = BaseInterface(
    resource_type="projects", s3_transfer_method=None, endpoint="v2/projects"
)

DraftInterface = BaseInterface(
    resource_type="drafts", s3_transfer_method=None, endpoint="v2/drafts"
)

FolderInterface = BaseInterface(resource_type="Folder", s3_transfer_method=None, endpoint="folders")
