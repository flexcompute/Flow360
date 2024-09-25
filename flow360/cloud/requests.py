"""Requests module"""

from typing import List, Optional, Union

import pydantic as pd_v2
import pydantic.v1 as pd
from typing_extensions import Literal

from flow360.flags import Flags

from ..component.flow360_params.flow360_params import Flow360MeshParams

LengthUnitType = Literal["m", "mm", "cm", "inch", "ft"]


###==== V1 API Payload definition ===###


class Flow360Requests(pd.BaseModel):
    """
    Represents a request for Flow360 WEBAPI.

    This class extends `pd.BaseModel` and provides a method for converting the request
    object into a dictionary representation.
    """

    def dict(self, *args, **kwargs) -> dict:
        """returns dict representation of request"""
        return super().dict(*args, by_alias=True, exclude_none=True, **kwargs)

    class Config:  # pylint: disable=too-few-public-methods
        """config"""

        allow_population_by_field_name = True


class NewSurfaceMeshRequest(Flow360Requests):
    """request for new surface mesh"""

    name: str = pd.Field()
    stem: str = pd.Field()
    tags: Optional[List[str]] = pd.Field()
    geometry_id: Optional[str] = pd.Field(alias="geometryId")
    config: Optional[str] = pd.Field()
    mesh_format: Optional[Literal["aflr3", "cgns", "stl"]] = pd.Field(alias="meshFormat")
    endianness: Optional[Literal["little", "big"]] = pd.Field(alias="meshEndianness")
    compression: Optional[Literal["gz", "bz2", "zst"]] = pd.Field(alias="meshCompression")
    solver_version: Optional[str] = pd.Field(alias="solverVersion")
    if Flags.beta_features():
        version: Optional[Literal["v1", "v2"]] = pd.Field(default="v1")


class NewVolumeMeshRequest(Flow360Requests):
    """request for new volume mesh"""

    name: str = pd.Field(alias="meshName")
    file_name: str = pd.Field(alias="fileName")
    tags: Optional[List[str]] = pd.Field(alias="meshTags")
    format: Literal["aflr3", "cgns"] = pd.Field(alias="meshFormat")
    endianness: Optional[Literal["little", "big"]] = pd.Field(alias="meshEndianness")
    compression: Optional[Literal["gz", "bz2", "zst"]] = pd.Field(alias="meshCompression")
    mesh_params: Optional[Flow360MeshParams] = pd.Field(alias="meshParams")
    solver_version: Optional[str] = pd.Field(alias="solverVersion")
    if Flags.beta_features():
        version: Optional[Literal["v1", "v2"]] = pd.Field(alias="version", default="v1")

    # pylint: disable=no-self-argument
    @pd.validator("mesh_params")
    def set_mesh_params(cls, value: Union[Flow360MeshParams, None]):
        """validate mesh params"""
        if value:
            return value.flow360_json()
        return value


class CopyExampleVolumeMeshRequest(Flow360Requests):
    """request for new volume mesh"""

    name: str = pd.Field()
    example_id: str = pd.Field(alias="meshId")


class NewFolderRequest(Flow360Requests):
    """request for new folder"""

    name: str = pd.Field()
    tags: Optional[List[str]] = pd.Field(alias="tags")
    parent_folder_id: Optional[str] = pd.Field(alias="parentFolderId", default="ROOT.FLOW360")
    type: Literal["folder"] = pd.Field("folder", const=True)


class MoveCaseItem(pd.BaseModel):
    """move case item"""

    id: str
    type: Literal["case"] = pd.Field("case", const=True)


class MoveFolderItem(pd.BaseModel):
    """move folder item"""

    id: str
    type: Literal["folder"] = pd.Field("folder", const=True)


class MoveToFolderRequest(Flow360Requests):
    """request for move to folder"""

    dest_folder_id: str = pd.Field(alias="destFolderId")
    items: List[Union[MoveCaseItem, MoveFolderItem]] = pd.Field()


###==== V2 API Payload definition ===###


class Flow360RequestsV2(pd_v2.BaseModel):
    """
    Represents a request for Flow360 WEBAPI.

    This class extends `pd.BaseModel` and provides a method for converting the request
    object into a dictionary representation.
    """

    def dict(self, *args, **kwargs) -> dict:
        """returns dict representation of request"""
        return super().dict(*args, by_alias=True, exclude_none=True, **kwargs)

    model_config = pd_v2.ConfigDict(populate_by_name=True)


class GeometryFileMeta(pd_v2.BaseModel):
    """[Simulation V2] File information for geometry."""

    name: str = pd_v2.Field(description="geometry file name")
    type: Literal["main", "dependency"] = pd_v2.Field(description="geometry hierarchy")


class NewGeometryRequest(Flow360RequestsV2):
    """[Simulation V2] Creates new project and a new geometry resource."""

    name: str = pd_v2.Field(description="project name")
    solver_version: str = pd_v2.Field(
        alias="solverVersion", description="solver version used for the project"
    )
    tags: List[str] = pd_v2.Field(default=[], description="project tags")
    files: List[GeometryFileMeta] = pd_v2.Field(description="list of files")
    parent_folder_id: str = pd_v2.Field(alias="parentFolderId", default="ROOT.FLOW360")
    length_unit: Literal["m", "mm", "cm", "inch", "ft"] = pd_v2.Field(
        alias="lengthUnit", description="project length unit"
    )
    description: str = pd_v2.Field(default="", description="project description")


class NewVolumeMeshRequestV2(Flow360RequestsV2):
    """[Simulation V2] Creates new project and a new volume mesh resource."""

    name: str = pd_v2.Field(description="project name")
    solver_version: str = pd_v2.Field(
        alias="solverVersion", description="solver version used for the project"
    )
    tags: List[str] = pd_v2.Field(default=[], description="project tags")
    file_name: str = pd_v2.Field(alias="fileName", description="file name of the volume mesh")
    parent_folder_id: str = pd_v2.Field(alias="parentFolderId", default="ROOT.FLOW360")
    length_unit: Literal["m", "mm", "cm", "inch", "ft"] = pd_v2.Field(
        alias="lengthUnit", description="project length unit"
    )
    description: str = pd_v2.Field(default="", description="project description")
    format: Literal["cgns", "aflr3"] = pd_v2.Field(description="data format")
