import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.primitives import GeometryBodyGroup, GhostSurface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock


def _minimal_geometry_entity_info():
    # Create minimal GeometryEntityInfo to satisfy GAI translator's compute_transformation_matrices
    info = GeometryEntityInfo(
        body_ids=["body00001"],
        body_attribute_names=["groupByFile"],
        grouped_bodies=[
            [
                GeometryBodyGroup(
                    name="group-file-1",
                    private_attribute_tag_key="groupByFile",
                    private_attribute_sub_components=["body00001"],
                )
            ]
        ],
        face_ids=[],
        face_attribute_names=[],
        grouped_faces=[[]],
        edge_ids=[],
        edge_attribute_names=[],
        grouped_edges=[[]],
    )
    with model_attribute_unlock(info, "body_group_tag"):
        info.body_group_tag = "groupByFile"
    return info


def test_surface_refinement_accepts_ghostsurface_and_kept_in_gai_json():
    with SI_unit_system:
        ghost = GhostSurface(name="symmetric")
        params = SimulationParams(
            meshing=MeshingParams(
                refinements=[
                    SurfaceRefinement(
                        entities=[ghost],
                        max_edge_length=0.1 * u.m,
                    )
                ]
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                project_entity_info=_minimal_geometry_entity_info(),
            ),
        )

    translated = get_surface_meshing_json(params, 1.0 * u.m)
    # In GAI mode, translator filters the params JSON; ensure the SurfaceRefinement with GhostSurface is present
    refinements = translated["meshing"]["refinements"]
    found = False
    for item in refinements:
        if item.get("refinement_type") == "SurfaceRefinement":
            entities = item.get("entities", {})
            stored = entities.get("stored_entities", [])
            if any(
                ent.get("private_attribute_entity_type_name") == "GhostSurface"
                and ent.get("name") == "symmetric"
                for ent in stored
            ):
                found = True
                break
    assert found
