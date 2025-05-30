"""Requests module"""

from datetime import datetime
from typing import Annotated, List, Optional, Union

import pydantic as pd_v2
import pydantic.v1 as pd
from pydantic.alias_generators import to_camel
from typing_extensions import Literal

from ..component.utils import is_valid_uuid

LengthUnitType = Literal["m", "mm", "cm", "inch", "ft"]


def _valid_id_validator(input_id: str):
    is_valid_uuid(input_id)
    return input_id


IDStringType = Annotated[str, pd_v2.AfterValidator(_valid_id_validator)]


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
        return super().model_dump(*args, by_alias=True, exclude_none=True, **kwargs)

    model_config = pd_v2.ConfigDict(populate_by_name=True, alias_generator=to_camel)


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


class NewSurfaceMeshRequestV2(Flow360RequestsV2):
    """[Simulation V2] Creates new project and a new surface mesh resource."""

    name: str = pd_v2.Field(description="project name")
    solver_version: str = pd_v2.Field(
        alias="solverVersion", description="solver version used for the project"
    )
    tags: List[str] = pd_v2.Field(default=[], description="project tags")
    parent_folder_id: str = pd_v2.Field(alias="parentFolderId", default="ROOT.FLOW360")
    length_unit: Literal["m", "mm", "cm", "inch", "ft"] = pd_v2.Field(
        alias="lengthUnit", description="project length unit"
    )
    description: str = pd_v2.Field(default="", description="project description")
    file_name: str = pd_v2.Field(alias="fileName", description="file name of the surface mesh")


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


class _Resource(Flow360RequestsV2):
    type: Literal["Case", "Project"]
    id: str


class NewReportRequest(Flow360RequestsV2):
    "New report request"
    name: str
    resources: List[_Resource]
    config_json: str
    solver_version: str


class DraftCreateRequest(Flow360RequestsV2):
    """Data model for draft create request"""

    name: Optional[str] = pd.Field(None)
    project_id: IDStringType = pd.Field()
    source_item_id: IDStringType = pd.Field()
    source_item_type: Literal[
        "Project", "Folder", "Geometry", "SurfaceMesh", "VolumeMesh", "Case", "Draft"
    ] = pd.Field()
    solver_version: str = pd.Field()
    fork_case: bool = pd.Field()
    interpolation_volume_mesh_id: Optional[str] = pd.Field(None)
    interpolation_case_id: Optional[str] = pd.Field(None)
    tags: Optional[List[str]] = pd.Field(None)

    @pd_v2.field_validator("name", mode="after")
    @classmethod
    def _generate_default_name(cls, values):
        if values is None:
            values = "Draft " + datetime.now().strftime("%m-%d %H:%M:%S")
        return values


class ForceCreationConfig(Flow360RequestsV2):
    """Data model for force creation configuration"""

    start_from: Literal["SurfaceMesh", "VolumeMesh", "Case"] = pd.Field()


class DraftRunRequest(Flow360RequestsV2):
    """Data model for draft run request"""

    up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"] = pd.Field()
    use_in_house: bool = pd.Field()
    use_gai: bool = pd.Field()
    force_creation_config: Optional[ForceCreationConfig] = pd.Field(
        None,
    )
    source_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh", "Case"] = pd.Field(
        exclude=True
    )

    @pd_v2.model_validator(mode="after")
    def _validate_force_creation_config(self):
        # pylint: disable=no-member

        order = {"Geometry": 0, "SurfaceMesh": 1, "VolumeMesh": 2, "Case": 3}
        source_order = order[self.source_item_type]
        up_to_order = order[self.up_to]

        if self.force_creation_config is not None:
            force_start_order = order[self.force_creation_config.start_from]
            if (force_start_order <= source_order and self.source_item_type != "Case") or (
                self.source_item_type == "Case" and self.force_creation_config.start_from != "Case"
            ):
                raise ValueError(
                    f"Invalid force creation configuration: 'start_from' ({self.force_creation_config.start_from}) "
                    f"must be later than 'source_item_type' ({self.source_item_type})."
                )

            if force_start_order > up_to_order:
                raise ValueError(
                    f"Invalid force creation configuration: 'start_from' ({self.force_creation_config.start_from}) "
                    f"cannot be later than 'up_to' ({self.up_to})."
                )
        return self
