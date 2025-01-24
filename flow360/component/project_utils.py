"""
Support class and functions for project interface.
"""

import datetime
from typing import List, Literal, Optional

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.component.utils import parse_datetime
from flow360.log import log


class AssetStatistics(pd.BaseModel):
    """Statistics for an asset"""

    count: int
    successCount: int
    runningCount: int
    divergedCount: int
    errorCount: int


class ProjectStatistics(pd.BaseModel):
    """Statistics for a project"""

    geometry: Optional[AssetStatistics] = pd.Field(None, alias="Geometry")
    surface_mesh: Optional[AssetStatistics] = pd.Field(None, alias="SurfaceMesh")
    volume_mesh: Optional[AssetStatistics] = pd.Field(None, alias="VolumeMesh")
    case: Optional[AssetStatistics] = pd.Field(None, alias="Case")


class ProjectInfo(pd.BaseModel):
    """Information about a project, retrieved from the projects get API"""

    name: str
    project_id: str = pd.Field(alias="id")
    tags: list[str] = pd.Field()
    description: str = pd.Field()
    statistics: ProjectStatistics = pd.Field()
    solver_version: Optional[str] = pd.Field(
        None, alias="solverVersion", description="If None then the project is from old database"
    )
    created_at: str = pd.Field(alias="createdAt")
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"] = pd.Field(
        alias="rootItemType"
    )

    @pd.computed_field
    @property
    def local_time_zone_created_time(self) -> datetime.datetime:
        """Convert string time to datetime obj and also convert to local time zone"""
        return parse_datetime(self.created_at)


class ProjectRecords(pd.BaseModel):
    """Holds all records of a user's project"""

    records: List[ProjectInfo] = pd.Field()

    def __str__(self):
        """Print out all info about the project"""
        if not self.records:
            output_str = "No matching projects found. Try skip naming patterns to show all."
            return output_str
        output_str = ">>> Projects sorted by creation time:\n"
        # pylint: disable=not-an-iterable
        for item in self.records:
            output_str += f" Name:         {item.name}\n"
            output_str += f" Created at:   {item.local_time_zone_created_time.strftime('%Y-%m-%d %H:%M %Z')}\n"
            output_str += f" Created with: {item.root_item_type}\n"
            output_str += f" ID:           {item.project_id}\n"
            output_str += (
                f" Link:         https://flow360.simulation.cloud/workbench/{item.project_id}\n"
            )
            if item.tags:
                output_str += f" Tags:         {item.tags}\n"
            if item.description:
                output_str += f" Description:  {item.description}\n"
            if item.statistics.geometry:
                output_str += f" Geometry count:     {item.statistics.geometry.count}\n"
            if item.statistics.surface_mesh:
                output_str += f" Surface Mesh count: {item.statistics.surface_mesh.count}\n"
            if item.statistics.volume_mesh:
                output_str += f" Volume Mesh count:  {item.statistics.volume_mesh.count}\n"
            if item.statistics.case:
                output_str += f" Case count:         {item.statistics.case.count}\n"

            output_str += "\n"
        return output_str


def show_projects_with_keyword_filter(search_keyword: str):
    """Show all projects with a keyword filter"""
    # pylint: disable=invalid-name
    MAX_DISPLAYABLE_ITEM_COUNT = 200
    MAX_SEARCHABLE_ITEM_COUNT = 1000
    _api = RestApi(ProjectInterface.endpoint, id=None)
    resp = _api.get(
        params={
            "page": "0",
            "size": MAX_SEARCHABLE_ITEM_COUNT,
            "filterKeywords": search_keyword,
            "sortFields": ["createdAt"],
            "sortDirections": ["asc"],
        }
    )

    all_projects = ProjectRecords.model_validate({"records": resp["records"]})
    log.info("%s", str(all_projects))

    if resp["total"] > MAX_DISPLAYABLE_ITEM_COUNT:
        log.warning(
            f"Total number of projects matching the keyword on the cloud is {resp['total']}, "
            f"but only the latest {MAX_DISPLAYABLE_ITEM_COUNT} will be displayed. "
        )
    log.info("Total number of matching projects on the cloud: %d", resp["total"])
