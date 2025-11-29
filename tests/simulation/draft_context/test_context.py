from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

from flow360.component.simulation.draft_context import DraftContext, create_draft
from flow360.component.surface_mesh_v2 import SurfaceMeshV2

SIMULATION_JSON_PATH = (
    Path(__file__).resolve().parent.parent / "service" / "data" / "simulation.json"
)


@lru_cache(maxsize=1)
def _load_simulation_dict() -> dict:
    with open(SIMULATION_JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_surface_mesh() -> SurfaceMeshV2:
    simulation_dict = copy.deepcopy(_load_simulation_dict())
    asset_cache = simulation_dict.get("private_attribute_asset_cache", {})
    entity_info = asset_cache.get("project_entity_info")
    if entity_info and "draft_entities" in entity_info:
        entity_info["draft_entities"] = []
    mesh = SurfaceMeshV2(id=None)
    mesh = SurfaceMeshV2._from_supplied_entity_info(simulation_dict, mesh)
    mesh.internal_registry = mesh._entity_info.get_registry(mesh.internal_registry)
    return mesh
