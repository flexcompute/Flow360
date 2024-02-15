"""webAPI interface definitions
"""

from typing import Union

from pydantic import BaseModel

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


FolderInterface = BaseInterface(resource_type="Folder", s3_transfer_method=None, endpoint="folders")
