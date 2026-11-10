"""
Lightweight project tree assembly helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

NodeT = TypeVar("NodeT")


def get_project_tree_parent_id(record: Mapping[str, Any]) -> str | None:
    """Return the effective parent ID for a project tree record."""

    return record.get("parentCaseId") or record.get("parentId")


def build_project_tree(
    records: Iterable[Mapping[str, Any]],
    *,
    create_node: Callable[[Mapping[str, Any]], NodeT],
    add_child: Callable[[NodeT, NodeT], None],
    strict: bool = False,
) -> tuple[NodeT | None, dict[str, NodeT]]:
    """Build a project tree from flat API records without SDK dependencies."""

    record_list = list(records)
    nodes: dict[str, NodeT] = {}
    records_by_id: dict[str, Mapping[str, Any]] = {}
    root_ids: list[str] = []

    for record in record_list:
        node_id = record["id"]
        if strict and node_id in nodes:
            raise ValueError(f"Project tree response contains duplicate item: {node_id}")
        nodes[node_id] = create_node(record)
        records_by_id[node_id] = record
        if get_project_tree_parent_id(record) is None:
            root_ids.append(node_id)

    for node_id, record in records_by_id.items():
        parent_id = get_project_tree_parent_id(record)
        if parent_id is None:
            continue
        if parent_id not in nodes:
            if strict:
                raise ValueError(
                    f"Project tree response references missing parent {parent_id} for {node_id}"
                )
            continue
        add_child(nodes[parent_id], nodes[node_id])

    if strict and len(root_ids) != 1:
        raise ValueError(f"Project tree response contains {len(root_ids)} root items")

    root_id = root_ids[-1] if root_ids else None
    return nodes.get(root_id), nodes
