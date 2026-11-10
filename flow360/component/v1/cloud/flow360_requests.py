"""V1 requests for cloud component"""

from typing import List, Optional, Union

import pydantic.v1 as pd
from typing_extensions import Literal

from flow360.cloud.flow360_requests import Flow360Requests
from flow360.component.v1.flow360_params import Flow360MeshParams
from flow360.flags import Flags


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
