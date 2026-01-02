"""Utility functions for web/cloud resource operations."""

from typing import List, Literal

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.exceptions import Flow360ValueError


class ProjectDependencyMetadata(pd.BaseModel):
    """Metadata of a project dependency resource."""

    resource_id: str = pd.Field(alias="id")
    name: str = pd.Field(alias="name")

    model_config = pd.ConfigDict(extra="ignore", validate_by_alias=True)


def get_project_dependency_resource_metadata(
    project_id: str, resource_type: Literal["Geometry", "SurfaceMesh"]
) -> List[ProjectDependencyMetadata]:
    """
    Fetch raw dependency resource data from cloud API.

    Parameters
    ----------
    project_id : str
        The project ID
    resource_type : Literal["Geometry", "SurfaceMesh"]
        The type of dependency resource to retrieve

    Returns
    -------
    List[dict]
        List of raw dependency resource dictionaries from the API response.
        Each dict contains at minimum 'id' and 'name' fields.

    Raises
    ------
    Flow360ValueError
        If resource_type is not supported
    """
    resp = RestApi(ProjectInterface.endpoint, id=project_id).get(method="dependency")

    if resource_type == "Geometry":
        return [ProjectDependencyMetadata(**item) for item in resp["geometryDependencyResources"]]
    if resource_type == "SurfaceMesh":
        return [
            ProjectDependencyMetadata(**item) for item in resp["surfaceMeshDependencyResources"]
        ]

    raise Flow360ValueError(f"Unsupported resource type: {resource_type}")
