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
) -> tuple[NodeT, dict[str, NodeT]]:
    """Build a project tree from flat API records without SDK dependencies."""

    record_list = list(records)
    nodes: dict[str, NodeT] = {}
    root_ids: list[str] = []

    for record in record_list:
        node_id = record["id"]
        if node_id in nodes:
            raise ValueError(f"Project tree response contains duplicate item: {node_id}")
        nodes[node_id] = create_node(record)

    for record in record_list:
        node_id = record["id"]
        parent_id = get_project_tree_parent_id(record)
        if parent_id is None:
            root_ids.append(node_id)
            continue
        if parent_id not in nodes:
            raise ValueError(
                f"Project tree response references missing parent {parent_id} for {node_id}"
            )
        add_child(nodes[parent_id], nodes[node_id])

    if len(root_ids) != 1:
        raise ValueError(f"Project tree response contains {len(root_ids)} root items")

    return nodes[root_ids[0]], nodes
