"""Tests for DraftContext.preview_unselected().

The point of this API is catching silent under-selection: a glob that matches nothing,
or a name whose capitalization does not match the geometry, leaves entities out of every
assignment without raising. These tests pin what counts as "in the comparison pool".
"""

import pytest
from flow360_schema.framework.entity.entity_selector import (
    BodyGroupSelector,
    SurfaceSelector,
)
from flow360_schema.models.entities.geometry_entities import GeometryBodyGroup
from flow360_schema.models.entities.surface_entities import (
    ImportedSurface,
    MirroredSurface,
    Surface,
)
from flow360_schema.models.entity_info import GeometryEntityInfo

import flow360 as fl
from flow360.component.simulation.draft_context.context import DraftContext

FACE_NAMES = ["body_main", "rim_FL", "rim_FR", "TYRE_FL", "wt_floor"]


def _surface(name):
    return Surface(
        name=name,
        private_attribute_id=name,
        private_attribute_sub_components=[name],
    )


@pytest.fixture()
def draft():
    """Draft holding five surfaces and one body group, no cloud resources involved."""
    entity_info = GeometryEntityInfo(
        face_ids=FACE_NAMES,
        face_attribute_names=["faceId"],
        face_group_tag="faceId",
        grouped_faces=[[_surface(name) for name in FACE_NAMES]],
        body_ids=["vehicle"],
        body_attribute_names=["groupByFile"],
        body_group_tag="groupByFile",
        grouped_bodies=[
            [
                GeometryBodyGroup(
                    name="vehicle",
                    private_attribute_id="vehicle",
                    private_attribute_tag_key="groupByFile",
                    private_attribute_sub_components=["vehicle"],
                )
            ]
        ],
        bodies_face_edge_ids={"vehicle": {"face_ids": FACE_NAMES}},
    )
    return DraftContext(entity_info=entity_info)


class TestPool:
    def test_reports_everything_a_selection_misses(self, draft):
        rims = SurfaceSelector(name="rims").match("rim_*")

        assert sorted(draft.preview_unselected(rims)) == [
            "TYRE_FL",
            "body_main",
            "wt_floor",
        ]

    def test_case_sensitive_glob_miss_is_reported(self, draft):
        """`*[tT]yre*` cannot match `TYRE_FL`; the surface must show up as unselected."""
        half_case_insensitive = SurfaceSelector(name="tyres").match("*[tT]yre*")

        assert half_case_insensitive.name and draft.preview_selector(half_case_insensitive) == []
        assert "TYRE_FL" in draft.preview_unselected(half_case_insensitive)

    def test_full_coverage_reports_nothing(self, draft):
        everything = SurfaceSelector(name="all").match("*")

        assert draft.preview_unselected(everything) == []

    def test_mirrored_surfaces_are_in_the_pool(self, draft):
        """Selectors can reach MirroredSurface, so a missed mirrored surface is reported."""
        mirrored = MirroredSurface(
            name="mirror_of_rim_FL", mirror_plane_id="plane-1", surface_id="rim_FL"
        )
        # pylint: disable=protected-access
        draft._entity_registry.fast_register(mirrored, set())

        rims = SurfaceSelector(name="rims").match("rim_*")
        assert "mirror_of_rim_FL" in draft.preview_unselected(rims)
        # A pattern that does reach it is not reported: pool and selector reach agree.
        assert draft.preview_unselected(SurfaceSelector(name="all").match("*")) == []

    def test_imported_surfaces_are_not_in_the_pool(self, draft):
        """ImportedSurface is deliberately not selectable, so it is not reported."""
        imported = ImportedSurface(name="probe", private_attribute_id="probe")
        # pylint: disable=protected-access
        draft._entity_registry.fast_register(imported, set())

        assert draft.preview_unselected(SurfaceSelector(name="all").match("*")) == []

    def test_body_group_selection_uses_the_body_group_pool(self, draft):
        """The pool follows the selection's entity kind, not just surfaces."""
        assert draft.preview_unselected(BodyGroupSelector(name="none").match("nomatch*")) == [
            "vehicle"
        ]


class TestSelectionForms:
    def test_selector_list_unions_coverage(self, draft):
        rims = SurfaceSelector(name="rims").match("rim_*")
        body = SurfaceSelector(name="body").match("body_*")

        assert sorted(draft.preview_unselected([rims, body])) == ["TYRE_FL", "wt_floor"]

    def test_mixed_entities_and_selectors(self, draft):
        rims = SurfaceSelector(name="rims").match("rim_*")
        explicit = draft.surfaces["body_main"]

        assert sorted(draft.preview_unselected([explicit, rims])) == ["TYRE_FL", "wt_floor"]

    def test_single_entity(self, draft):
        unselected = draft.preview_unselected(draft.surfaces["body_main"])

        assert "body_main" not in unselected
        assert len(unselected) == len(FACE_NAMES) - 1

    def test_entity_list_from_boundary_condition(self, draft):
        """A Wall's assignment can be handed over verbatim to audit what it leaves out."""
        wall = fl.Wall(name="wall", surfaces=[draft.surfaces[name] for name in FACE_NAMES[:2]])

        assert sorted(draft.preview_unselected(wall.entities)) == [
            "TYRE_FL",
            "rim_FR",
            "wt_floor",
        ]

    def test_registry_view(self, draft):
        assert draft.preview_unselected(draft.surfaces) == []

    def test_return_instances(self, draft):
        unselected = draft.preview_unselected(
            SurfaceSelector(name="rims").match("rim_*"), return_names=False
        )

        assert all(isinstance(entity, Surface) for entity in unselected)
        assert sorted(entity.name for entity in unselected) == [
            "TYRE_FL",
            "body_main",
            "wt_floor",
        ]


class TestAmbiguousSelections:
    def test_empty_selection_raises(self, draft):
        with pytest.raises(Exception, match="cannot tell which kind of entity"):
            draft.preview_unselected([])

    def test_mixed_entity_kinds_raise(self, draft):
        mixed = [
            SurfaceSelector(name="rims").match("rim_*"),
            BodyGroupSelector(name="bodies").match("*"),
        ]
        with pytest.raises(Exception, match="single entity kind"):
            draft.preview_unselected(mixed)


def test_beta_notice_shown_once_per_draft(draft, capsys):
    """These get called in loops; one notice per feature per draft, not per call."""
    selector = SurfaceSelector(name="rims").match("rim_*")
    for _ in range(3):
        draft.preview_unselected(selector)
        draft.preview_selector(selector)

    output = capsys.readouterr().out
    assert output.count("beta feature") == 2
    assert "preview_unselected" in output
    assert "preview_selector" in output
