"""
Case component
"""
from __future__ import annotations

import json
import math

from pydantic import Extra, Field

from ..cloud.http_util import http
from ..cloud.s3_utils import S3TransferType
from .flow360_base_model import (
    Flow360BaseModel,
    Flow360Resource,
    before_submit_only,
    is_object_cloud_resource,
)
from .flow360_solver_params import Flow360Params
from .utils import is_valid_uuid


class CaseMeta(Flow360BaseModel, extra=Extra.allow):
    """
    Case component
    """

    id: str = Field(alias="caseId")
    case_mesh_id: str = Field(alias="caseMeshId")
    status: str = Field(alias="caseStatus")
    parent_id: str = Field(alias="parentId")


# pylint: disable=too-many-instance-attributes
class Case(Flow360Resource):
    """
    Case component
    """

    def __init__(self, case_id: str = None):
        super().__init__(
            resource_type="Case",
            info_type_class=CaseMeta,
            s3_transfer_method=S3TransferType.CASE,
            endpoint="cases",
            id=case_id,
        )
        if case_id is not None:
            self.get_info()
            self._params = Flow360Params(**json.loads(self.get(method="runtimeParams")["content"]))

        self.other_case = None
        self.parent_case = None
        self.parent_id = None
        self.tags = None

    def __str__(self):
        if self._info is not None:
            return self.info.__str__()
        return "Case is not yet submitted"

    @property
    def params(self):
        """
        returns case params
        """
        return self._params

    @params.setter
    @before_submit_only
    def params(self, value):
        """
        sets case params (before submit only)
        """
        if not isinstance(value, Flow360Params):
            raise ValueError("params are not of type Flow360Params.")
        self._params = value

    @property
    def name(self):
        """
        returns case name
        """
        if self.is_cloud_resource():
            return self.info.name
        return self._name

    @name.setter
    @before_submit_only
    def name(self, value):
        """
        sets case name (before submit only)
        """
        self._name = value

    @property
    def volume_mesh_id(self):
        """
        returns volume mesh id (before submit only)
        """
        if self.is_cloud_resource():
            return self.info.case_mesh_id
        return self._volume_mesh_id

    @volume_mesh_id.setter
    @before_submit_only
    def volume_mesh_id(self, value):
        """
        sets volume mesh id (before submit only)
        """
        self._volume_mesh_id = value

    @before_submit_only
    def submit(self):
        """
        submits case to cloud for running
        """
        assert self.name
        assert self.volume_mesh_id or self.other_case or self.parent_id or self.parent_case
        assert self.params

        self.validate_case_inputs(pre_submit_checks=True)

        volume_mesh_id = self.volume_mesh_id
        parent_id = self.parent_id
        if parent_id is not None:
            self.parent_case = Case(self.parent_id)

        if self.parent_case is not None:
            parent_id = self.parent_case.id
            volume_mesh_id = self.parent_case.volume_mesh_id

        volume_mesh_id = volume_mesh_id or self.other_case.volume_mesh_id

        is_valid_uuid(volume_mesh_id)
        is_valid_uuid(parent_id, ignore_none=True)

        resp = self.post(
            json={
                "name": self.name,
                "meshId": volume_mesh_id,
                "runtimeParams": self.params.json(),
                "tags": self.tags,
                "parentId": parent_id,
            },
            path=f"volumemeshes/{volume_mesh_id}/case",
        )
        self._info = CaseMeta(**resp)
        self.init_id(self._info.id)

    def copy(self, name: str = None, params: Flow360Params = None, tags: [str] = None) -> Case:
        """
        Alias for retry case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return self.retry(name, params, tags)

    def retry(self, name: str = None, params: Flow360Params = None, tags: [str] = None) -> Case:
        """
        Retry case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        name = name or self.name or self.info.name
        params = params or self.params.copy(deep=True)
        new_case = Case.new(name, params, other_case=self, tags=tags)
        return new_case

    def continuation(
        self, name: str = None, params: Flow360Params = None, tags: [str] = None
    ) -> Case:
        """
        Alias for fork a case to continue simulation
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return self.fork(name, params, tags)

    def fork(self, name: str = None, params: Flow360Params = None, tags: [str] = None) -> Case:
        """
        Fork a case to continue simulation
        :param name:
        :param params:
        :param tags:
        :return:
        """

        name = name or self.name or self.info.name
        params = params or self.params.copy(deep=True)
        return Case.new(name, params, parent_case=self, tags=tags)

    @before_submit_only
    def validate_case_inputs(self, pre_submit_checks=False):
        """
        validates case inputs (before submit only)
        """
        if self.volume_mesh_id is not None and self.other_case is not None:
            raise ValueError("You cannot specify both volume_mesh_id AND other_case.")

        if self.parent_id is not None and self.parent_case is not None:
            raise ValueError("You cannot specify both parent_id AND parent_case.")

        if self.parent_id is not None or self.parent_case is not None:
            if self.volume_mesh_id is not None or self.other_case is not None:
                raise ValueError(
                    "You cannot specify volume_mesh_id OR other_case when parent case provided."
                )

        is_valid_uuid(self.volume_mesh_id, ignore_none=True)

        if pre_submit_checks:
            is_object_cloud_resource(self.other_case)
            is_object_cloud_resource(self.parent_case)

    @classmethod
    def from_cloud(cls, case_id: str):
        """
        get case from cloud
        """
        return cls(case_id)

    # pylint: disable=too-many-arguments
    @classmethod
    def new(
        cls,
        name: str,
        params: Flow360Params,
        volume_mesh_id: str = None,
        tags: [str] = None,
        parent_id=None,
        other_case: Case = None,
        parent_case: Case = None,
    ) -> Case:
        """
        Create new case
        :param name:
        :param params:
        :param volume_mesh_id:
        :param other_case:
        :param tags:
        :param parent_id:
        :param parent_case:
        :return:
        """

        assert name
        assert volume_mesh_id or other_case or parent_id or parent_case
        assert params

        if not isinstance(params, Flow360Params):
            raise ValueError("params are not of type Flow360Params.")

        new_case = cls()
        new_case.name = name
        new_case.volume_mesh_id = volume_mesh_id
        new_case.other_case = other_case
        new_case.params = params.copy(deep=True)
        new_case.tags = tags
        new_case.parent_id = parent_id
        new_case.parent_case = parent_case

        new_case.validate_case_inputs()

        return new_case

    # pylint: disable=too-many-arguments
    @classmethod
    def submit_multiple_phases(
        cls,
        name: str,
        volume_mesh_id: str,
        params: Flow360Params,
        tags: [str] = None,
        phase_steps=1,
    ):
        """
        Create multiple cases from volume mesh
        :param name:
        :param volume_mesh_id:
        :param params:
        :param tags:
        :param parent_id:
        :param phase_steps:
        :return:
        """

        assert name
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
                    "name": f"{name}_{i}",
                    "meshId": volume_mesh_id,
                    "runtimeParams": params.json(),
                    "tags": tags,
                    "parentId": parent_id,
                },
            )

            result.append(cls(**case))

        return result
