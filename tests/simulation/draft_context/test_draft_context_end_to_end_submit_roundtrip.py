import json

import pytest

import flow360.component.simulation.units as u
from flow360.component.project import create_draft
from flow360.component.project_utils import (
    set_up_params_for_uploading,
    validate_params_with_context,
)
from flow360.component.simulation.framework.entity_selector import SurfaceSelector
from flow360.component.simulation.meshing_param.meshing_specs import MeshingDefaults
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import PorousMedium
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import Box, Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.web.draft import Draft


def _find_all_selector_lists(obj) -> list[list]:
    """Collect all list values found at dict key 'selectors' in a nested structure."""
    found: list[list] = []
    stack = [obj]
    visited: set[int] = set()
    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)
        if isinstance(cur, dict):
            if "selectors" in cur and isinstance(cur["selectors"], list):
                found.append(cur["selectors"])
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


def _get_entity_list_dict(container_dict: dict) -> dict:
    """Return the nested entity list dict from a model/output dict (handles 'entities' vs alias keys)."""
    for key in ("entities", "surfaces", "volumes"):
        if key in container_dict and isinstance(container_dict[key], dict):
            return container_dict[key]
    raise KeyError("Expected an EntityList dict under one of: entities/surfaces/volumes.")


@pytest.mark.usefixtures("mock_response")
def test_draft_end_to_end_selector_and_draft_entity_roundtrip(mock_surface_mesh, monkeypatch):
    """
    End-to-end test mimicking user workflow:
    - create_draft() provides entity_info context
    - user builds SimulationParams using selectors (reused in multiple places) + draft entities (reused)
    - pre-submit: set_up_params_for_uploading() strips selector-matched stored_entities
    - local validation: validate_model()
    - upload: draft.update_simulation_params() tokenizes selectors + populates used_selectors
    - backend: validate_model() again to ensure roundtrip and that materialization links objects
    """
    with create_draft(new_run_from=mock_surface_mesh) as draft:
        with SI_unit_system:
            # Shared selector reused in multiple places (same selector_id).
            shared_selector = SurfaceSelector(name="sel_shared_fuselage").match("fuselage")
            selector_id = shared_selector.selector_id

            # Persistent entity picked explicitly to create overlap with selector selection.
            explicit_surface = draft.surfaces["fuselage"]

            # Shared draft entity reused in multiple places (draft entity types).
            shared_box = Box(
                name="shared_box",
                center=(0, 0, 0) * u.m,
                size=(1, 1, 1) * u.m,
            )

            params = SimulationParams(
                # Draft entity used in meshing refinement
                meshing=MeshingParams(
                    refinements=[
                        UniformRefinement(
                            spacing=0.1 * u.m,
                            entities=[shared_box],
                        )
                    ],
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=1e-5 * u.m,
                    ),
                ),
                # Selector used in a boundary model + porous medium uses the same draft entity
                models=[
                    Wall(
                        name="wall_fuselage",
                        # Overlap: explicit stored entity + selector selecting the same surface.
                        entities=[explicit_surface, shared_selector],
                    ),
                    PorousMedium(
                        entities=[shared_box],
                        darcy_coefficient=(1e6, 0, 0) / u.m**2,
                        forchheimer_coefficient=(1, 0, 0) / u.m,
                    ),
                ],
                # Selector reused in outputs as well
                outputs=[
                    SurfaceOutput(name="out1", entities=[shared_selector], output_fields=["Cp"]),
                ],
            )

        # Verify overlap exists pre-setup (stored_entities contains the explicit surface).
        wall_model = next(m for m in params.models if isinstance(m, Wall))
        assert [s.name for s in wall_model.entities.stored_entities] == ["fuselage"]
        assert wall_model.entities.selectors is not None
        assert wall_model.entities.selectors[0].selector_id == selector_id

        # 3.1 Mimic project pre-submit: set up params and strip selector-matched stored_entities.
        params = set_up_params_for_uploading(
            params=params,
            root_asset=mock_surface_mesh,
            length_unit=1 * u.m,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )
        wall_model = next(m for m in params.models if isinstance(m, Wall))
        assert wall_model.entities.stored_entities == []

        # Local validation stage (validate_model should pass).
        params, errors = validate_params_with_context(
            params=params, root_item_type="SurfaceMesh", up_to="VolumeMesh"
        )
        assert errors is None

        # Mimic upload by calling Draft.update_simulation_params() but without any Draft submit API.
        uploaded_payload: dict = {}

        def _capture_post(self, *, json=None, method=None, **_kwargs):
            uploaded_payload["json"] = json
            uploaded_payload["method"] = method
            return {}

        monkeypatch.setattr(Draft, "post", _capture_post, raising=True)

        Draft(draft_id="00000000-0000-0000-0000-000000000000").update_simulation_params(params)

        assert uploaded_payload["method"] == "simulation/file"
        uploaded_dict = json.loads(uploaded_payload["json"]["data"])

        # Success criteria (a) proper selector token + (c) proper used_selectors population.
        used_selectors = uploaded_dict["private_attribute_asset_cache"]["used_selectors"]
        assert isinstance(used_selectors, list)
        assert len(used_selectors) == 1
        assert used_selectors[0]["selector_id"] == selector_id

        # All selectors lists in the payload must be token lists.
        selector_lists = _find_all_selector_lists(uploaded_dict)
        assert selector_lists, "Expected selectors lists to exist in uploaded payload."
        assert all(lst == [selector_id] for lst in selector_lists)

        # Success criteria (b): stored_entities should not contain entities implied by selector.
        wall_dict = next(m for m in uploaded_dict["models"] if m.get("type") == "Wall")
        wall_entities_dict = _get_entity_list_dict(wall_dict)
        assert wall_entities_dict["stored_entities"] == []
        assert wall_entities_dict["selectors"] == [selector_id]

        # 3.2 Mimic backend validation service roundtrip.
        validated, backend_errors, _ = validate_model(
            params_as_dict=uploaded_dict,
            validated_by=ValidationCalledBy.SERVICE,
            root_item_type="SurfaceMesh",
            validation_level=None,
        )
        assert backend_errors is None

        # Validate expansion works (a successful validate_model is usually sufficient; we still assert names).
        wall_model_v2 = next(m for m in validated.models if isinstance(m, Wall))
        # Create a DraftContext from validated params to test selector preview
        from flow360.component.simulation.draft_context import DraftContext

        entity_info = validated.private_attribute_asset_cache.project_entity_info
        temp_draft = DraftContext(entity_info=entity_info)
        wall_selector = wall_model_v2.entities.selectors[0]
        selected_names = temp_draft.preview_selector(wall_selector, return_names=True)
        assert selected_names == ["fuselage"]

        # Assert selector materialization links instances across references (same object).
        surface_outputs = [o for o in validated.outputs if isinstance(o, SurfaceOutput)]
        assert len(surface_outputs) >= 1
        wall_sel = wall_model_v2.entities.selectors[0]
        assert surface_outputs[0].entities.selectors[0] is wall_sel
        assert validated.private_attribute_asset_cache.used_selectors[0] is wall_sel

        # Assert draft entity materialization links reused draft entity objects (same object).
        refinement_box = validated.meshing.refinements[0].entities.stored_entities[0]
        porous_model = next(m for m in validated.models if isinstance(m, PorousMedium))
        porous_box = porous_model.entities.stored_entities[0]
        assert refinement_box is porous_box
