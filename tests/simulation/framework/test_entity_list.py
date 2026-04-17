import flow360 as fl
from flow360.component.simulation.draft_context import DraftContext
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.entity_selector import SurfaceSelector
from flow360.component.simulation.primitives import Surface


def _build_preview_context(boundary_names: list[str]):
    with fl.SI_unit_system:
        boundaries = [
            Surface(name=name, private_attribute_id=f"{name}_id") for name in boundary_names
        ]
    entity_info = SurfaceMeshEntityInfo(boundaries=boundaries)
    draft = DraftContext(entity_info=entity_info)
    return boundaries, draft


def test_preview_selection_returns_names_by_default():
    _, draft = _build_preview_context(["tail", "wing_leading", "wing_trailing"])
    selector = SurfaceSelector(name="wing_surfaces").match("wing*")

    previewed_names = draft.preview_selector(selector)

    assert previewed_names == ["wing_leading", "wing_trailing"]


def test_preview_selection_returns_instances_when_requested():
    _, draft = _build_preview_context(["body00001", "body00002"])
    selector = SurfaceSelector(name="second_body").match("body00002")

    expanded_entities = draft.preview_selector(selector, return_names=False)

    assert [entity.name for entity in expanded_entities] == ["body00002"]
    assert all(isinstance(entity, Surface) for entity in expanded_entities)
