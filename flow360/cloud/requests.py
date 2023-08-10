"""Requests module"""

from typing import List, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from ..component.flow360_params.flow360_params import Flow360MeshParams


class NewVolumeMeshRequest(pd.BaseModel):
    """request for new volume mesh"""

    name: str = pd.Field(alias="meshName")
    file_name: str = pd.Field(alias="fileName")
    tags: Optional[List[str]] = pd.Field(alias="meshTags")
    format: Literal["aflr3", "cgns"] = pd.Field(alias="meshFormat")
    endianness: Optional[Literal["little", "big"]] = pd.Field(alias="meshEndianness")
    compression: Optional[Literal["gz", "bz2", "zst"]] = pd.Field(alias="meshCompression")
    mesh_params: Optional[Flow360MeshParams] = pd.Field(alias="meshParams")
    solver_version: Optional[str] = pd.Field(alias="solverVersion")

    # pylint: disable=no-self-argument
    @pd.validator("mesh_params")
    def set_mesh_params(cls, value: Union[Flow360MeshParams, None]):
        """validate mesh params"""
        if value:
            return value.to_flow360_json()
        return value

    def dict(self, *args, **kwargs) -> dict:
        """returns dict representation of request"""
        return super().dict(*args, by_alias=True, exclude_none=True, **kwargs)

    class Config:  # pylint: disable=too-few-public-methods
        """config"""

        allow_population_by_field_name = True
