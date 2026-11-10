import pytest

from flow360.component.simulation.web.project_tree import (
    build_project_tree,
    get_project_tree_parent_id,
)


def _build_dict_tree(records, *, strict=False):
    def create_node(record):
        return {"id": record["id"], "children": []}

    def add_child(parent, child):
        parent["children"].append(child)

    return build_project_tree(
        records,
        create_node=create_node,
        add_child=add_child,
        strict=strict,
    )


def test_get_project_tree_parent_id_prefers_parent_case_id():
    assert (
        get_project_tree_parent_id({"parentCaseId": "case-parent", "parentId": "vm-parent"})
        == "case-parent"
    )
    assert (
        get_project_tree_parent_id({"parentCaseId": None, "parentId": "vm-parent"}) == "vm-parent"
    )


def test_build_project_tree_uses_case_parent_edges():
    root, nodes = _build_dict_tree(
        [
            {"id": "geo-1", "parentId": None, "parentCaseId": None},
            {"id": "vm-1", "parentId": "geo-1", "parentCaseId": None},
            {"id": "case-1", "parentId": "vm-1", "parentCaseId": None},
            {"id": "case-2", "parentId": "vm-1", "parentCaseId": "case-1"},
        ]
    )

    assert root["id"] == "geo-1"
    assert nodes["vm-1"]["children"][0]["id"] == "case-1"
    assert nodes["case-1"]["children"][0]["id"] == "case-2"


def test_build_project_tree_rejects_invalid_records():
    with pytest.raises(ValueError, match="duplicate item"):
        _build_dict_tree(
            [
                {"id": "geo-1", "parentId": None, "parentCaseId": None},
                {"id": "geo-1", "parentId": None, "parentCaseId": None},
            ],
            strict=True,
        )

    with pytest.raises(ValueError, match="missing parent"):
        _build_dict_tree(
            [{"id": "case-1", "parentId": "vm-1", "parentCaseId": None}],
            strict=True,
        )

    with pytest.raises(ValueError, match="2 root items"):
        _build_dict_tree(
            [
                {"id": "geo-1", "parentId": None, "parentCaseId": None},
                {"id": "geo-2", "parentId": None, "parentCaseId": None},
            ],
            strict=True,
        )


def test_build_project_tree_default_preserves_sdk_leniency():
    root, nodes = _build_dict_tree(
        [
            {"id": "orphan", "parentId": "missing", "parentCaseId": None},
            {"id": "geo-1", "parentId": None, "parentCaseId": None},
            {"id": "geo-2", "parentId": None, "parentCaseId": None},
            {"id": "geo-2", "parentId": None, "parentCaseId": None},
        ]
    )

    assert root["id"] == "geo-2"
    assert nodes["orphan"]["children"] == []
    assert set(nodes) == {"orphan", "geo-1", "geo-2"}
