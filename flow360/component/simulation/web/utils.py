"""Utility functions for web/cloud resource operations."""

import re
from typing import List, Literal, Tuple

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.exceptions import Flow360ValueError


def _get_pydantic_major_minor_version() -> Tuple[int, int]:
    """Extract Pydantic (major, minor) from the installed version string."""

    pydantic_version_module = getattr(pd, "version", None)
    version_string = getattr(pydantic_version_module, "VERSION", None)
    if version_string is None:
        version_string = getattr(pd, "__version__", "")

    match = re.match(r"^\s*(\d+)\.(\d+)", str(version_string))
    if match is None:
        return (0, 0)

    return (int(match.group(1)), int(match.group(2)))


def _is_pydantic_version_greater_or_equal_to_2_11() -> bool:
    """Return True if current pydantic version is >= 2.11 (major/minor)."""

    return _get_pydantic_major_minor_version() >= (2, 11)


_PROJECT_DEPENDENCY_METADATA_CONFIG_KWARGS = {"extra": "ignore"}
if _is_pydantic_version_greater_or_equal_to_2_11():
    _PROJECT_DEPENDENCY_METADATA_CONFIG_KWARGS["validate_by_alias"] = True
else:
    _PROJECT_DEPENDENCY_METADATA_CONFIG_KWARGS["populate_by_name"] = True


class ProjectDependencyMetadata(pd.BaseModel):
    """Metadata of a project dependency resource."""

    resource_id: str = pd.Field(alias="id")
    name: str = pd.Field(alias="name")

    model_config = pd.ConfigDict(**_PROJECT_DEPENDENCY_METADATA_CONFIG_KWARGS)


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
