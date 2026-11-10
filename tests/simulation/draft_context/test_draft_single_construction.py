"""Single-construction guarantees of create_draft / DraftContext.

create_draft hands the asset's fresh entity_info deserialization straight to the
draft (no second deep copy); DraftContext optionally accepts a prebuilt registry
built from that same instance instead of rebuilding one.
"""

import flow360_schema.models.simulation.units as u
from flow360_schema.framework.entity.entity_registry import EntityRegistry
from flow360_schema.models.entities.volume_entities import Box

from flow360.component.project import create_draft
from flow360.component.simulation.draft_context.context import DraftContext


def test_draft_mutation_does_not_leak_into_asset_entity_info(mock_surface_mesh):
    asset_entity_info = mock_surface_mesh._entity_info
    asset_draft_count = len(asset_entity_info.draft_entities)
    asset_boundary_ids = [b.private_attribute_id for b in asset_entity_info.boundaries]

    with create_draft(new_run_from=mock_surface_mesh) as draft:
        assert draft._entity_info is not asset_entity_info
        draft_surface = draft.surfaces["fuselage"]
        assert all(draft_surface is not boundary for boundary in asset_entity_info.boundaries)

        draft._entity_info.draft_entities.append(
            Box.from_principal_axes(
                name="leak-probe",
                center=(0, 0, 0) * u.m,
                size=(1, 1, 1) * u.m,
                axes=((1, 0, 0), (0, 1, 0)),
            )
        )

    assert len(asset_entity_info.draft_entities) == asset_draft_count
    assert [b.private_attribute_id for b in asset_entity_info.boundaries] == asset_boundary_ids


def test_prebuilt_registry_is_used_without_rebuild(mock_surface_mesh, monkeypatch):
    entity_info = mock_surface_mesh.entity_info
    prebuilt_registry = EntityRegistry.from_entity_info(entity_info)

    def _fail_rebuild(*args, **kwargs):
        raise AssertionError("DraftContext must not rebuild a registry when one is passed in")

    monkeypatch.setattr(EntityRegistry, "from_entity_info", _fail_rebuild)
    draft = DraftContext(entity_info=entity_info, entity_registry=prebuilt_registry)

    assert draft._entity_registry is prebuilt_registry
    with draft:
        surface = draft.surfaces["fuselage"]
    assert any(surface is boundary for boundary in entity_info.boundaries)


def test_prebuilt_pair_behaves_like_self_built(mock_surface_mesh):
    entity_info = mock_surface_mesh.entity_info
    paired = DraftContext(
        entity_info=entity_info, entity_registry=EntityRegistry.from_entity_info(entity_info)
    )
    self_built = DraftContext(entity_info=mock_surface_mesh.entity_info)

    with paired:
        paired_names = sorted(s.name for s in paired.surfaces["*"])
    with self_built:
        self_built_names = sorted(s.name for s in self_built.surfaces["*"])
    assert paired_names == self_built_names
