import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.primitives import (
    GeometryBodyGroup,
    GhostCircularPlane,
    GhostSurface,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    SurfaceRefinement_to_faces,
    get_surface_meshing_json,
)
from flow360.component.simulation.translator.utils import (
    translate_setting_and_apply_to_all_entities,
)
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock


def _minimal_geometry_entity_info():
    # Create minimal GeometryEntityInfo to satisfy GAI translator's body-group matrix injection
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
        face_ids=["body00001_face00001"],
        face_attribute_names=["faceId"],
        grouped_faces=[
            [Surface(name="wall", private_attribute_sub_components=["body00001_face00001"])]
        ],
        edge_ids=[],
        edge_attribute_names=[],
        grouped_edges=[[]],
    )
    with model_attribute_unlock(info, "body_group_tag"):
        info.body_group_tag = "groupByFile"
    with model_attribute_unlock(info, "face_group_tag"):
        info.face_group_tag = "faceId"
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


def test_surface_refinement_uses_boundary_name_for_ghost_circular_plane_with_beta_mesher():
    with SI_unit_system:
        ghost = GhostCircularPlane(name="symmetric")
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_max_edge_length=0.2 * u.m,
                    curvature_resolution_angle=15 * u.deg,
                    surface_edge_growth_rate=1.1,
                ),
                refinements=[
                    SurfaceRefinement(
                        entities=[ghost],
                        max_edge_length=0.1 * u.m,
                    )
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=True,
                project_entity_info=_minimal_geometry_entity_info(),
            ),
        )

    translated = get_surface_meshing_json(params, 1.0 * u.m)

    assert translated["faces"]["symmetric"] == {
        "maxEdgeLength": pytest.approx(0.1),
        "curvatureResolutionAngle": pytest.approx(15.0),
    }
    assert translated["faces"]["body00001_face00001"] == {"maxEdgeLength": pytest.approx(0.2)}
    assert translated["boundaries"]["body00001_face00001"] == {"boundaryName": "wall"}


def test_surface_refinement_does_not_fallback_for_empty_sub_component_list():
    with SI_unit_system:
        refinement = SurfaceRefinement(
            entities=[Surface(name="wall", private_attribute_sub_components=[])],
            max_edge_length=0.1 * u.m,
        )

    translated = translate_setting_and_apply_to_all_entities(
        [refinement],
        SurfaceRefinement,
        translation_func=SurfaceRefinement_to_faces,
        translation_func_global_max_edge_length=0.2 * u.m,
        translation_func_global_curvature_resolution_angle=15 * u.deg,
        use_sub_item_as_key=True,
    )

    assert translated == {}
