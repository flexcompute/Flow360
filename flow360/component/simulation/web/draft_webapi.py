"""
Thin draft web API wrapper.
"""

from __future__ import annotations

import json

from flow360.cloud.rest_api import RestApi
from flow360.cloud.flow360_requests import DraftCreateRequest
from flow360.component.interfaces import DraftInterface


class DraftWebApi:
    """Thin wrapper around draft endpoints."""

    def __init__(self, draft_id: str):
        self.draft_id = draft_id
        self._api = RestApi(DraftInterface.endpoint, id=draft_id)

    @staticmethod
    def _unwrap_data(response):
        if isinstance(response, dict) and "data" in response:
            return response["data"]
        return response

    def get_info(self):
        return self._unwrap_data(self._api.get())

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        source_item_id: str,
        source_item_type: str,
        solver_version: str,
        fork_case: bool,
        name: str | None = None,
        interpolation_volume_mesh_id: str | None = None,
        interpolation_case_id: str | None = None,
        tags: list[str] | None = None,
    ):
        api = RestApi(DraftInterface.endpoint)
        payload = DraftCreateRequest(
            name=name,
            project_id=project_id,
            source_item_id=source_item_id,
            source_item_type=source_item_type,
            solver_version=solver_version,
            fork_case=fork_case,
            interpolation_volume_mesh_id=interpolation_volume_mesh_id,
            interpolation_case_id=interpolation_case_id,
            tags=tags,
        ).model_dump(by_alias=True)
        return cls._unwrap_data(api.post(json=payload))

    @classmethod
    def list_records(cls, project_id: str):
        api = RestApi(DraftInterface.endpoint)
        response = api.get(params={"projectId": project_id})
        return response.get("records", [])

    def get_simulation_json(self):
        response = self._api.get(method="simulation/file", params={"type": "simulation"})
        if isinstance(response, dict) and "simulationJson" in response:
            return response["simulationJson"]
        return response

    def set_simulation_json(self, simulation_json):
        if not isinstance(simulation_json, str):
            simulation_json = json.dumps(simulation_json)

        return self._api.post(
            method="simulation/file",
            json={
                "data": simulation_json,
                "type": "simulation",
                "version": "",
            },
        )

    def run(
        self,
        up_to: str,
        *,
        use_in_house: bool = False,
        use_gai: bool = False,
        start_from: str | None = None,
        job_type: str | None = None,
        priority: int | None = None,
    ):
        payload = {
            "upTo": up_to,
            "useInHouse": use_in_house,
            "useGai": use_gai,
        }
        if start_from is not None:
            payload["forceCreationConfig"] = {"startFrom": start_from}
        if job_type is not None:
            payload["jobType"] = job_type
        if priority is not None:
            payload["priority"] = priority

        return self._unwrap_data(self._api.post(method="run", json=payload))

    def get(self, path=None, method=None, json=None, params=None):
        return self._api.get(path=path, method=method, json=json, params=params)

    def post(self, json, path=None, method=None):
        return self._api.post(json=json, path=path, method=method)

    def put(self, json, path=None, method=None):
        return self._api.put(json=json, path=path, method=method)

    def patch(self, json, path=None, method=None):
        return self._api.patch(json=json, path=path, method=method)

    def delete(self, path=None, method=None):
        return self._api.delete(path=path, method=method)
