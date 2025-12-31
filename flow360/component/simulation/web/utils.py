"""Utility functions for web/cloud resource operations."""

from typing import List, Literal

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.exceptions import Flow360ValueError


def get_project_dependency_resources_raw(
    project_id: str, resource_type: Literal["Geometry", "SurfaceMesh"]
) -> List[dict]:
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
        return resp["geometryDependencyResources"]
    if resource_type == "SurfaceMesh":
        return resp["surfaceMeshDependencyResources"]

    raise Flow360ValueError(f"Unsupported resource type: {resource_type}")
