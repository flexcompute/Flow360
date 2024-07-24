"""Requests module"""

from typing import List, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from flow360.flags import Flags

from ..component.flow360_params.flow360_params import Flow360MeshParams

length_unit_type = Literal["m", "mm", "cm", "inch", "ft"]


class Flow360Requests(pd.BaseModel):
    """
    Represents a request for Flow360 WEBAPI.

    This class extends `pd.BaseModel` and provides a method for converting the request
    object into a dictionary representation.
    """

    def dict(self, *args, **kwargs) -> dict:
        """returns dict representation of request"""
        return super().model_dump(*args, by_alias=True, exclude_none=True, **kwargs)

    class Config:  # pylint: disable=too-few-public-methods
        """config"""

        populate_by_name = True


class GeometryFileMeta(pd.BaseModel):
    name: str = pd.Field(description="geometry file name")
    type: Literal["main", "dependency"] = pd.Field(description="geometry hierarchy")


class NewGeometryRequest(Flow360Requests):
    """Creates new project and a new geometry resource."""

    name: str = pd.Field(description="project name")
    solver_version: str = pd.Field(
        alias="solverVersion", description="solver version used for the project"
    )
    tags: List[str] = pd.Field(default=[], description="project tags")
    files: List[GeometryFileMeta] = pd.Field(description="list of files")
    parent_folder_id: str = pd.Field(alias="parentFolderId", default="ROOT.FLOW360")
    length_unit: Literal["m", "mm", "cm", "inch", "ft"] = pd.Field(
        alias="lengthUnit", description="project length unit"
    )
    description: str = pd.Field(default="", description="project description")
