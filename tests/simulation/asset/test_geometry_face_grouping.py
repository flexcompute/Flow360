"""Offline unit tests for the rule-based face/entity selection and grouping API
on Geometry (see flow360/component/geometry_face_grouping.py).

These run fully locally: a bare ``Geometry(id=None)`` (see AssetBase -- "When id
is None, the asset is meant to be operating in local mode") with a hand-built
entity_info and a hand-built topology tree injected directly, so no network
call (and no real CAD import) is involved.
"""

import pytest
from flow360_schema.geometry.selection import root
from flow360_schema.geometry.selection_evaluation import models, resolve_selection
from flow360_schema.models.entity_info import parse_entity_info_model

from flow360.component.geometry import Geometry


def _ids(geometry, selection) -> list:
    """Test shorthand for the separated evaluation: descriptions (built on the
    `root` token) are instance-agnostic; the tree is an explicit argument."""
    return resolve_selection(selection, geometry.topology_tree)


RED = [255, 0, 0]  # tree-side colorRGB (the topology tree stores 0-255 ints)
GREEN = [0, 255, 0]
RED_CSS = "#FF0000"  # rule-side color= (rules take CSS color strings)
GREEN_CSS = "#00FF00"


def _geometry(tree: dict, face_ids: list) -> Geometry:
    geometry = Geometry(id=None)
    geometry._entity_info = parse_entity_info_model(
        {
            "type_name": "GeometryEntityInfo",
            "face_ids": face_ids,
            "face_attribute_names": [],
            "grouped_faces": [[]],
        }
    )
    geometry._topology_tree = tree
    return geometry


@pytest.fixture
def cad_geometry() -> Geometry:
    """A 2-part, 2-body CADGeometry tree (mirrors a STEP-via-V2-importer tree):
    assembly -> {wing, fuselage} parts, each with one body and two faces (one red,
    one green)."""
    tree = {
        "type": "GeometryResource",
        "children": [
            {
                "type": "ModelFile",
                "name": "parts.step",
                "modelingKernel": "CADGeometry",
                "children": [
                    {
                        "type": "Assembly",
                        "uuid": "asm-1",
                        "name": "asm1",
                        "children": [
                            {
                                "type": "Part",
                                "uuid": "part-1",
                                "name": "wing",
                                "attributes": {"__STEP/ProductName": "wing"},
                                "children": [
                                    {
                                        "type": "SolidBody",
                                        "uuid": "body-1",
                                        "name": "wing_body",
                                        "material": {"name": "Aluminum 6061"},
                                        "children": [
                                            {
                                                "type": "Face",
                                                "uuid": "body1_face1",
                                                "name": "top",
                                                "colorRGB": RED,
                                            },
                                            {
                                                "type": "Face",
                                                "uuid": "body1_face2",
                                                "name": "bottom",
                                                "colorRGB": GREEN,
                                            },
                                        ],
                                    }
                                ],
                            },
                            {
                                "type": "Part",
                                "uuid": "part-2",
                                "name": "fuselage",
                                "children": [
                                    {
                                        "type": "SolidBody",
                                        "uuid": "body-2",
                                        "name": "fuselage_body",
                                        "children": [
                                            {
                                                "type": "Face",
                                                "uuid": "body2_face1",
                                                "colorRGB": RED,
                                            },
                                            {
                                                "type": "Face",
                                                "uuid": "body2_face2",
                                                "colorRGB": GREEN,
                                            },
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }
    return _geometry(tree, ["body1_face1", "body1_face2", "body2_face1", "body2_face2"])


@pytest.fixture
def egads_geometry() -> Geometry:
    """A single-body EGADSGeometry tree (flat: ModelFile -> SolidBody -> Face, no
    Assembly/Part level)."""
    tree = {
        "type": "GeometryResource",
        "children": [
            {
                "type": "ModelFile",
                "name": "sphere.egads",
                "modelingKernel": "EGADSGeometry",
                "children": [
                    {
                        "type": "SolidBody",
                        "uuid": "body-1",
                        "name": "body00001",
                        "children": [
                            {"type": "Face", "uuid": "body00001_face00001"},
                            {"type": "Face", "uuid": "body00001_face00002"},
                        ],
                    }
                ],
            }
        ],
    }
    return _geometry(tree, ["body00001_face00001", "body00001_face00002"])


@pytest.fixture
def binary_stl_geometry() -> Geometry:
    """A Binary STL ModelFile: the ModelFile itself is the lone anonymous groupable
    leaf, with no navigable hierarchy at all."""
    tree = {
        "type": "GeometryResource",
        "children": [
            {
                "type": "ModelFile",
                "name": "part.stl",
                "modelingKernel": "BinarySTL",
                "uuid": "part.stl_body",
            }
        ],
    }
    return _geometry(tree, ["part.stl_body"])


# ---- navigation chains -------------------------------------------------------


def test_children_chain(cad_geometry):
    sel = _ids(cad_geometry, root.children().children().children().children(color=RED_CSS))
    assert set(sel) == {"body1_face1", "body2_face1"}


def test_descendants_red_faces(cad_geometry):
    assert set(_ids(cad_geometry, root.descendants(color=RED_CSS))) == {
        "body1_face1",
        "body2_face1",
    }


def test_bodies_chain(cad_geometry):
    assert set(_ids(cad_geometry, root.bodies().faces(color=RED_CSS))) == {
        "body1_face1",
        "body2_face1",
    }


def test_named_part_chain(cad_geometry):
    assert set(_ids(cad_geometry, root.parts(name="wing").faces(color=RED_CSS))) == {"body1_face1"}


def test_assemblies_and_bodies_nodes(cad_geometry):
    assert set(_ids(cad_geometry, root.assemblies())) == {"asm-1"}
    assert set(_ids(cad_geometry, root.bodies())) == {"body-1", "body-2"}
    assert set(_ids(cad_geometry, root.parts())) == {"part-1", "part-2"}


# ---- conditions ---------------------------------------------------------------


def test_color_condition_hex(cad_geometry):
    assert set(_ids(cad_geometry, root.faces(color="#FF0000"))) == {"body1_face1", "body2_face1"}


def test_color_condition_rgb_function(cad_geometry):
    assert set(_ids(cad_geometry, root.faces(color="rgb(255, 0, 0)"))) == {
        "body1_face1",
        "body2_face1",
    }


def test_color_condition_list_rejected(cad_geometry):
    with pytest.raises(ValueError, match="CSS color string"):
        root.faces(color=RED)


def test_name_condition(cad_geometry):
    assert set(_ids(cad_geometry, root.faces(name="top"))) == {"body1_face1"}


def test_material_condition(cad_geometry):
    assert set(_ids(cad_geometry, root.bodies(material="*6061*"))) == {"body-1"}


def test_attributes_condition(cad_geometry):
    assert set(_ids(cad_geometry, root.parts(attributes={"__STEP/ProductName": "wing"}))) == {
        "part-1"
    }


def test_unknown_match_field_raises(cad_geometry):
    with pytest.raises(ValueError, match="Unknown match field"):
        _ids(cad_geometry, root.faces(body="body-1"))


def test_empty_match_returns_empty(cad_geometry):
    assert len(_ids(cad_geometry, root.bodies(name="not_existing"))) == 0


# ---- union / difference --------------------------------------------------------


def test_union_covers_all_faces(cad_geometry):
    all_faces = {"body1_face1", "body1_face2", "body2_face1", "body2_face2"}
    union = root.faces(color=RED_CSS) | root.faces(color=GREEN_CSS)
    assert set(_ids(cad_geometry, union)) == all_faces


def test_difference_is_green(cad_geometry):
    diff = root.faces() - root.faces(color=RED_CSS)
    assert set(_ids(cad_geometry, diff)) == {"body1_face2", "body2_face2"}


def test_plus_is_rejected(cad_geometry):
    from flow360_schema.geometry.selection import RuleSyntaxError

    with pytest.raises(RuleSyntaxError, match="Selections are sets"):
        _ = root.faces(color=RED_CSS) + root.faces(color=GREEN_CSS)


# ---- group registration (draft-state model) ---------------------------------


def test_registered_groups_resolve_by_bracket_access(cad_geometry):
    """Groups registered into an entity_info resolve via ['<name>'] once the
    rule-based tag is active — what Wall(surfaces=[...]) needs on a draft."""
    from flow360.component.geometry_face_grouping import register_group_assignments

    register_group_assignments(
        cad_geometry._entity_info,  # the draft's working copy in local mode
        {"body1_face1": "red_group", "body2_face1": "red_group"},
    )
    cad_geometry.group_faces_by_tag("rule-based-grouping")
    assert set(cad_geometry["red_group"].private_attribute_sub_components) == {
        "body1_face1",
        "body2_face1",
    }


def test_register_group_assignments_rejects_unknown_face_ids(cad_geometry):
    """Topology-tree uuids and entity_info face ids come from separate backend
    artifacts — a resolved id missing from entity_info must fail loudly, not
    surface later as a broken Surface at meshing."""
    from flow360.component.geometry_face_grouping import register_group_assignments
    from flow360.exceptions import Flow360ValueError

    with pytest.raises(Flow360ValueError, match="face-id mismatch"):
        register_group_assignments(
            cad_geometry._entity_info,
            {"body1_face1": "ok", "ghost_face": "ok"},
        )


def _two_file_geometry() -> Geometry:
    """Two ModelFiles, two faces each — the multi-model-file fixture."""
    tree = {
        "type": "GeometryResource",
        "children": [
            {
                "type": "ModelFile",
                "name": "a.step",
                "modelingKernel": "CADGeometry",
                "children": [
                    {
                        "type": "SolidBody",
                        "uuid": "body-a",
                        "children": [
                            {"type": "Face", "uuid": "a_face1"},
                            {"type": "Face", "uuid": "a_face2"},
                        ],
                    }
                ],
            },
            {
                "type": "ModelFile",
                "name": "b.step",
                "modelingKernel": "CADGeometry",
                "children": [
                    {
                        "type": "SolidBody",
                        "uuid": "body-b",
                        "children": [
                            {"type": "Face", "uuid": "b_face1"},
                            {"type": "Face", "uuid": "b_face2"},
                        ],
                    }
                ],
            },
        ],
    }
    return _geometry(tree, ["a_face1", "a_face2", "b_face1", "b_face2"])


def test_file_scoped_groups_materialize_on_a_two_file_tree():
    """One group per model file is the accepted spelling on a multi-file geometry:
    each group becomes a Surface under its own name, nothing derived."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.geometry_face_grouping import apply_face_grouping_rules

    geometry = _two_file_geometry()
    rules = FaceGroupingRules(
        groups=[
            FaceGroupRule(name="wall_a", rule=root.model_files(name="a.step").faces()),
            FaceGroupRule(name="wall_b", rule=root.model_files(name="b.step").faces()),
        ]
    )
    apply_face_grouping_rules(rules, geometry.topology_tree, geometry._entity_info)
    geometry.group_faces_by_tag("rule-based-grouping")
    assert set(geometry["wall_a"].private_attribute_sub_components) == {"a_face1", "a_face2"}
    assert set(geometry["wall_b"].private_attribute_sub_components) == {"b_face1", "b_face2"}


def test_preview_and_create_draft_reject_groups_left_without_faces_alike(cad_geometry):
    """A rule group left with no faces -- its rule matches nothing, or an earlier group
    claimed every face it matches -- is refused by the read-only preview and by the
    materializing path alike, from the one schema check, so a preview never shows a
    grouping that create_draft would then reject."""
    from types import SimpleNamespace

    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.geometry_face_grouping import apply_face_grouping_rules
    from flow360.component.simulation.draft_context.context import DraftContext

    rules = FaceGroupingRules(
        groups=[
            FaceGroupRule(name="wing", rule=root.parts(name="wing").faces()),
            FaceGroupRule(name="wing_red", rule=root.parts(name="wing").faces(color=RED_CSS)),
            FaceGroupRule(name="ghost", rule=root.faces(name="no_such_face*")),
            FaceGroupRule(name="rest", is_remaining=True),
        ]
    )
    draft_like = SimpleNamespace(_topology_tree_provider=lambda: cad_geometry.topology_tree)
    with pytest.raises(ValueError, match="left with no faces") as previewed:
        DraftContext.preview_face_grouping(draft_like, rules)
    with pytest.raises(ValueError, match="left with no faces") as applied:
        apply_face_grouping_rules(rules, cad_geometry.topology_tree, cad_geometry._entity_info)
    assert str(previewed.value) == str(applied.value)
    assert "'wing_red'" in str(applied.value) and "'ghost'" in str(applied.value)
    # nothing materialized on the way to the error
    assert "rule-based-grouping" not in cad_geometry.entity_info.face_attribute_names


def test_empty_remaining_group_is_allowed(cad_geometry):
    """An isRemaining catch-all left with zero faces is legitimate (all faces
    matched by rules) — only rule-bearing groups must match something."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.geometry_face_grouping import apply_face_grouping_rules

    rules = FaceGroupingRules(
        groups=[
            FaceGroupRule(name="everything", rule=root.faces()),
            FaceGroupRule(name="leftovers", is_remaining=True),
        ]
    )
    tag = apply_face_grouping_rules(rules, cad_geometry.topology_tree, cad_geometry._entity_info)
    assert tag == "rule-based-grouping"
    # all faces claimed by the rule; the empty catch-all produced no group
    cad_geometry.group_faces_by_tag("rule-based-grouping")
    assert set(cad_geometry["everything"].private_attribute_sub_components) == {
        "body1_face1",
        "body1_face2",
        "body2_face1",
        "body2_face2",
    }


def test_load_face_grouping_rules_reads_the_asset_cache(cad_geometry):
    """The loading half: rules stored in the source asset's simulation.json asset
    cache come back as a payload; an asset carrying none yields None."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.project import _load_face_grouping_rules

    rules = FaceGroupingRules(groups=[FaceGroupRule(name="all", rule=root.faces())])
    cad_geometry._simulation_dict_cache_for_local_mode = {
        "private_attribute_asset_cache": {"face_grouping_rules": rules.model_dump(mode="json")}
    }
    assert _load_face_grouping_rules(cad_geometry) == rules

    cad_geometry._simulation_dict_cache_for_local_mode = {"private_attribute_asset_cache": {}}
    assert _load_face_grouping_rules(cad_geometry) is None


def test_face_grouping_precedence_between_explicit_and_inherited(cad_geometry):
    """The deciding half, with no asset in sight: inherited rules become the draft's
    rule state on a geometry root, an explicit payload replaces them, an explicit tag
    string over inherited rules is ambiguous and raises, and with nothing inherited the
    caller's choice passes through untouched."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.project import _select_face_grouping
    from flow360.exceptions import Flow360ValueError

    inherited = FaceGroupingRules(groups=[FaceGroupRule(name="all", rule=root.faces())])
    override = FaceGroupingRules(groups=[FaceGroupRule(name="mine", rule=root.faces())])

    assert _select_face_grouping(None, inherited, cad_geometry) == (inherited, None)
    assert _select_face_grouping(override, inherited, cad_geometry) == (override, None)
    with pytest.raises(Flow360ValueError, match="silently discard"):
        _select_face_grouping("faceName", inherited, cad_geometry)

    assert _select_face_grouping("faceName", None, cad_geometry) == ("faceName", None)
    assert _select_face_grouping(None, None, cad_geometry) == (None, None)


def test_mesh_asset_carries_inherited_rules_as_metadata():
    """On mesh assets the resolved grouping is baked into the entity_info, and there is
    no topology tree to re-resolve against — inherited rules ride along as metadata."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.project import _select_face_grouping

    rules = FaceGroupingRules(groups=[FaceGroupRule(name="all", rule=root.faces())])

    class _MeshAssetStub:  # non-Geometry: the routing is by asset type
        pass

    assert _select_face_grouping(None, rules, _MeshAssetStub()) == (None, rules)


def test_modeling_kernel_property(cad_geometry):
    assert cad_geometry.modeling_kernel == "CADGeometry"


def test_models_helper(cad_geometry):
    handles = models(cad_geometry.topology_tree)
    assert list(handles.keys()) == ["parts.step"]
    assert len(_ids(cad_geometry, handles["parts.step"].faces())) == 4


def test_face_ids_property(cad_geometry):
    assert set(cad_geometry.face_ids) == {
        "body1_face1",
        "body1_face2",
        "body2_face1",
        "body2_face2",
    }


def test_show_topology_tree_contains_color_summary(cad_geometry):
    out = cad_geometry.show_topology_tree()
    assert "face colors" in out
    assert "#FF0000" in out  # the spelling a rule's color= takes


def test_topology_tree_is_the_envelope_the_endpoint_returns():
    """GET v2/geometries/<id>/topology-tree returns the published topologyTree.json as
    text; the tree is its envelope's `topologyTree`, fetched once."""
    import json
    from types import SimpleNamespace

    tree = {"type": "GeometryResource", "children": []}
    calls = []

    def get(method):
        calls.append(method)
        envelope = {"version": "2.0.0", "topologyTree": tree}
        return {"topologyTreeData": json.dumps(envelope), "downloadUrl": None}

    geometry = Geometry(id=None)
    geometry._webapi = SimpleNamespace(get=get)
    assert geometry.topology_tree == tree
    assert geometry.topology_tree == tree
    assert calls == ["topology-tree"]


def test_geometry_without_a_topology_tree_fails_by_name():
    """The endpoint answers null when no tree was written: an error naming the geometry,
    not a None that later surfaces as a confusing resolution failure."""
    from types import SimpleNamespace

    from flow360.exceptions import Flow360ValueError

    geometry = Geometry(id=None)
    geometry._webapi = SimpleNamespace(
        get=lambda method: {"topologyTreeData": None, "downloadUrl": None}
    )
    with pytest.raises(Flow360ValueError, match="has no topology tree"):
        geometry.topology_tree


# ---- kernel-driven structural restrictions ---------------------------------


def test_cad_has_no_zones(cad_geometry):
    with pytest.raises(ValueError, match="CAD has no zones"):
        _ids(cad_geometry, root.zones())


def test_egads_has_no_assemblies(egads_geometry):
    with pytest.raises(ValueError, match="EGADS has no assemblies"):
        _ids(egads_geometry, root.assemblies())


def test_egads_has_no_parts(egads_geometry):
    with pytest.raises(ValueError, match="EGADS has no parts"):
        _ids(egads_geometry, root.parts())


def test_egads_bodies_and_faces_work(egads_geometry):
    assert len(_ids(egads_geometry, root.bodies())) == 1
    assert len(_ids(egads_geometry, root.faces())) == 2


def test_binary_stl_has_no_children(binary_stl_geometry):
    with pytest.raises(ValueError, match="Binary STL file has no children nodes"):
        _ids(binary_stl_geometry, root.children())


def test_binary_stl_has_no_assemblies(binary_stl_geometry):
    with pytest.raises(ValueError, match="Binary STL model has no assemblies"):
        _ids(binary_stl_geometry, root.assemblies())


def test_register_group_assignments_targets_draft_copy_not_geometry(cad_geometry):
    """Rule-based grouping is draft state: registration materializes groups into an
    ISOLATED entity_info copy; the Geometry cloud asset stays immutable."""
    from flow360_schema.geometry.selection_rules import (
        FaceGroupingRules,
        FaceGroupRule,
        resolve_group_assignments,
    )

    from flow360.component.geometry_face_grouping import register_group_assignments

    rules = FaceGroupingRules(
        groups=[
            FaceGroupRule(name="wing_faces", rule=root.parts(name="wing").faces()),
            FaceGroupRule(name="rest", is_remaining=True),
        ]
    )
    assignment = resolve_group_assignments(rules, cad_geometry.topology_tree)
    assert assignment == {
        "body1_face1": "wing_faces",
        "body1_face2": "wing_faces",
        "body2_face1": "rest",
        "body2_face2": "rest",
    }

    # Isolated snapshot via model_dump + deserialize (master's entity_info property
    # already returns a fresh deserialization per access; project_utils no longer
    # ships a deep_copy_entity_info helper).
    draft_entity_info = type(cad_geometry.entity_info).deserialize(
        cad_geometry.entity_info.model_dump(mode="json")
    )
    register_group_assignments(draft_entity_info, assignment)

    assert "rule-based-grouping" in draft_entity_info.face_attribute_names
    groups = draft_entity_info.grouped_faces[
        draft_entity_info.face_attribute_names.index("rule-based-grouping")
    ]
    assert {s.name for s in groups} == {"wing_faces", "rest"}
    # the cloud asset's own entity_info is untouched
    assert "rule-based-grouping" not in cad_geometry.entity_info.face_attribute_names


def test_draft_context_exposes_face_grouping_state():
    """DraftContext transports the rules as draft state (persisted automatically
    at submission inside simulation.json's asset cache — no per-feature save
    method). The resolved groups live in the draft entity_info; there is no
    separate assignment state. Full construction needs a cloud-shaped
    entity_info; covered by the Nexus smoke."""
    import inspect

    from flow360.component.simulation.draft_context.context import DraftContext

    parameters = inspect.signature(DraftContext.__init__).parameters
    assert "face_grouping_rules" in parameters
    assert "topology_tree_provider" in parameters
    assert "face_grouping_assignment" not in parameters
    # One public accessor only.
    assert isinstance(DraftContext.face_grouping_rules, property)
    assert not hasattr(DraftContext, "face_grouping_assignment")


def test_preview_face_grouping_is_read_only_resolution(cad_geometry):
    """draft.preview_face_grouping(): resolve rules against the draft's topology
    tree without materializing anything — the iterate-before-committing tool."""
    from types import SimpleNamespace

    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.simulation.draft_context.context import DraftContext
    from flow360.exceptions import Flow360RuntimeError

    rules = FaceGroupingRules(
        groups=[
            FaceGroupRule(name="wing", rule=root.parts(name="wing").faces()),
            FaceGroupRule(name="rest", is_remaining=True),
        ]
    )
    draft_like = SimpleNamespace(_topology_tree_provider=lambda: cad_geometry.topology_tree)
    preview = DraftContext.preview_face_grouping(draft_like, rules)
    assert preview == {
        "wing": ["body1_face1", "body1_face2"],
        "rest": ["body2_face1", "body2_face2"],
    }
    # nothing materialized: the geometry's entity_info is untouched
    assert "rule-based-grouping" not in cad_geometry.entity_info.face_attribute_names

    # mesh-rooted drafts carry no tree
    with pytest.raises(Flow360RuntimeError, match="topology tree"):
        DraftContext.preview_face_grouping(SimpleNamespace(_topology_tree_provider=None), rules)


def test_rule_grouping_refuses_multi_geometry_drafts():
    """A multi-geometry draft merges the dependencies' faces into entity_info, but
    rules only ever see the primary geometry's tree — those faces would silently end
    up in no group at all, so the combination is refused. Dependencies are active by
    default on a multi-geometry project, not just via include_geometries."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.project import _reject_rule_grouping_on_multi_geometry
    from flow360.exceptions import Flow360ValueError

    rules = FaceGroupingRules(groups=[FaceGroupRule(name="all", rule=root.faces())])

    # single-geometry draft: nothing to reject
    _reject_rule_grouping_on_multi_geometry(rules, {})
    # a tag string stays legal on multi-geometry drafts
    _reject_rule_grouping_on_multi_geometry("faceName", {"geo-2": object()})

    with pytest.raises(Flow360ValueError, match="does not support multi-geometry"):
        _reject_rule_grouping_on_multi_geometry(rules, {"geo-2": object()})


def test_multi_geometry_refusal_counts_only_the_additional_components(mock_geometry, monkeypatch):
    """Merging entity info adds the root geometry to the draft's components, so
    create_draft refuses before merging: one included geometry is one additional
    component, not two."""
    from types import SimpleNamespace

    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.project import Project, create_draft
    from flow360.exceptions import Flow360ValueError

    monkeypatch.setattr(
        Project,
        "from_cloud",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(geometry=mock_geometry)),
    )
    included = SimpleNamespace(
        id="geo-included", name="included", entity_info=mock_geometry.entity_info
    )
    rules = FaceGroupingRules(groups=[FaceGroupRule(name="all", rule=root.faces())])
    with pytest.raises(Flow360ValueError, match=r"\(1 additional geometry component\(s\) active\)"):
        create_draft(new_run_from=mock_geometry, include_geometries=[included], face_grouping=rules)


def test_preview_and_create_draft_reject_cross_file_groups_alike():
    """A group whose faces land in two model files is refused at resolution — by the
    read-only preview and by the materializing path alike, from the one schema check —
    so a preview never shows a grouping that create_draft would then reject."""
    from types import SimpleNamespace

    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    from flow360.component.geometry_face_grouping import apply_face_grouping_rules
    from flow360.component.simulation.draft_context.context import DraftContext

    geometry = _two_file_geometry()
    rules = FaceGroupingRules(groups=[FaceGroupRule(name="wall", rule=root.faces())])
    draft_like = SimpleNamespace(_topology_tree_provider=lambda: geometry.topology_tree)
    with pytest.raises(ValueError, match=r"'wall': a.step \(2 face\(s\).*b.step \(2 face\(s\)"):
        DraftContext.preview_face_grouping(draft_like, rules)
    with pytest.raises(ValueError, match="more than one model file is not supported"):
        apply_face_grouping_rules(rules, geometry.topology_tree, geometry._entity_info)
    # nothing materialized on the way to the error
    assert "rule-based-grouping" not in geometry.entity_info.face_attribute_names


def test_face_grouping_namespace_exposes_the_authoring_surface():
    """`fl.face_grouping` is the discoverable home of rule authoring: the same objects
    the schema defines, no second import path to know about."""
    from flow360_schema.geometry.selection_rules import FaceGroupingRules, FaceGroupRule

    import flow360 as fl

    assert fl.face_grouping.root is root
    assert fl.face_grouping.FaceGroupRule is FaceGroupRule
    assert fl.face_grouping.FaceGroupingRules is FaceGroupingRules
    assert {"root", "node", "FaceGroupRule", "FaceGroupingRules"} <= set(fl.face_grouping.__all__)
