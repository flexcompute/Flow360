"""
Surface mesh component
"""
import math

from pydantic import Extra, Field

from flow360.cloud.http_util import http
from flow360.component.flow360_base_model import Flow360BaseModel
from flow360.component.flow360_solver_params import Flow360Params


class Case(Flow360BaseModel, extra=Extra.allow):
    """
    Case component
    """

    case_id: str = Field(alias="caseId")
    case_mesh_id: str = Field(alias="caseMeshId")
    status: str = Field(alias="caseStatus")
    parent_id: str = Field(alias="parentId")

    @classmethod
    def from_cloud(cls, case_id: str):
        """
        Get case info from cloud
        :param case_id:
        :return:
        """
        case = http.get(f"cases/{case_id}")
        return cls(**case)

    # pylint: disable=too-many-arguments
    @classmethod
    def submit_from_volume_mesh(
        cls,
        case_name: str,
        volume_mesh_id: str,
        params: Flow360Params,
        tags: [str] = None,
        parent_id=None,
    ):
        """
        Create case from volume mesh
        :param case_name:
        :param volume_mesh_id:
        :param params:
        :param tags:
        :param parent_id:
        :return:
        """

        assert case_name
        assert volume_mesh_id
        assert params

        case = http.post(
            f"volumemeshes/{volume_mesh_id}/case",
            json={
                "name": case_name,
                "meshId": volume_mesh_id,
                "runtimeParams": params.json(),
                "tags": tags,
                "parentId": parent_id,
            },
        )
        return cls(**case)

    # pylint: disable=too-many-arguments
    @classmethod
    def submit_multiple_phases(
        cls,
        case_name: str,
        volume_mesh_id: str,
        params: Flow360Params,
        tags: [str] = None,
        phase_steps=1,
    ):
        """
        Create multiple cases from volume mesh
        :param case_name:
        :param volume_mesh_id:
        :param params:
        :param tags:
        :param parent_id:
        :param phase_steps:
        :return:
        """

        assert case_name
        assert volume_mesh_id
        assert params
        assert phase_steps >= 1

        result = []

        total_steps = (
            params.time_stepping.max_physical_steps
            if params.time_stepping and params.time_stepping.max_physical_steps
            else 1
        )

        num_cases = math.ceil(total_steps / phase_steps)
        for i in range(1, num_cases + 1):
            parent_id = result[-1].case_id if result else None
            case = http.post(
                f"volumemeshes/{volume_mesh_id}/case",
                json={
                    "name": f"{case_name}_{i}",
                    "meshId": volume_mesh_id,
                    "runtimeParams": params.json(),
                    "tags": tags,
                    "parentId": parent_id,
                },
            )

            result.append(cls(**case))

        return result
