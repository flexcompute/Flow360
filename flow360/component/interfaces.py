"""webAPI interface definitions"""

from typing import Union

from pydantic import BaseModel

from flow360.cloud.s3_utils import S3TransferType


# pylint: disable=R0801
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

SurfaceMeshInterfaceV2 = BaseInterface(
    resource_type="Surface Mesh",
    s3_transfer_method=S3TransferType.SURFACE_MESH,
    endpoint="v2/surface-meshes",
)


VolumeMeshInterface = BaseInterface(
    resource_type="Volume Mesh",
    s3_transfer_method=S3TransferType.VOLUME_MESH,
    endpoint="volumemeshes",
)

VolumeMeshInterfaceV2 = BaseInterface(
    resource_type="Volume Mesh",
    s3_transfer_method=S3TransferType.VOLUME_MESH,
    endpoint="v2/volume-meshes",
)

CaseInterface = BaseInterface(
    resource_type="Case", s3_transfer_method=S3TransferType.CASE, endpoint="cases"
)

CaseInterfaceV2 = BaseInterface(
    resource_type="Case", s3_transfer_method=S3TransferType.CASE, endpoint="v2/cases"
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

FolderInterfaceV2 = BaseInterface(
    resource_type="Folder", s3_transfer_method=None, endpoint="v2/folders"
)

ReportInterface = BaseInterface(
    resource_type="Report",
    s3_transfer_method=S3TransferType.REPORT,
    endpoint="v2/report",
)
