"""UVF manifest parsing and vertex extraction.

Provides pure functions to parse UVF manifest JSON and extract face vertices
from binary tessellation data. No I/O — callers supply manifest and bin data.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Union

import numpy as np


def parse_manifest(manifest_data: Union[str, bytes]) -> List[dict]:
    """Parse raw manifest JSON into a list of manifest entries."""
    return json.loads(manifest_data)


def extract_face_vertices(  # pylint: disable=too-many-locals
    manifest: List[dict],
    face_ids: List[str],
    bin_data_map: Dict[str, bytes],
    lod_level: Optional[int] = None,
) -> np.ndarray:
    """Extract merged vertex positions for the given face IDs.

    Reads the UVF manifest structure to locate each face's vertex data within
    the binary buffers, supporting both indexed and unindexed geometry.

    Args:
        manifest: Parsed UVF manifest entries (list of dicts with "id", "resources", etc.).
        face_ids: Face IDs to extract vertices for.
        bin_data_map: Mapping of bin filename to raw bytes, as referenced by manifest entries.
        lod_level: LOD level override. None means use the manifest default.

    Returns:
        (N, 3) float32 array of vertex positions.

    Raises:
        KeyError: If any face_id is not found in the manifest.
        ValueError: If a solid body has no position section in its binary buffer.
    """
    by_id = {entry["id"]: entry for entry in manifest}

    missing = [fid for fid in face_ids if fid not in by_id]
    if missing:
        raise KeyError(f"face_ids not found in manifest: {missing}")

    # Group faces by packed parent to avoid re-parsing the same bin per parent
    parent_faces: Dict[str, List[dict]] = {}
    for fid in face_ids:
        face = by_id[fid]
        parent_id = face["attributions"]["packedParentId"]
        parent_faces.setdefault(parent_id, []).append(face)

    all_vertices: List[np.ndarray] = []

    for parent_id, faces in parent_faces.items():
        solid = by_id[parent_id]
        buffer_entry = resolve_buffers(solid, lod_level)
        bin_path = buffer_entry["path"]
        bin_data = bin_data_map[bin_path]

        sections = buffer_entry["sections"]
        index_section = _find_section(sections, "indices")
        position_section = _find_section(sections, "position")

        if position_section is None:
            raise ValueError(f"solid '{parent_id}' has no position section in bin")

        position_array = np.frombuffer(
            bin_data,
            dtype=np.float32,
            offset=position_section["offset"],
            count=position_section["length"] // 4,
        ).reshape(-1, 3)

        if index_section is not None:
            # Indexed geometry: face buffer locations reference into an index array
            index_array = np.frombuffer(
                bin_data,
                dtype=np.uint32,
                offset=index_section["offset"],
                count=index_section["length"] // 4,
            )
            for face in faces:
                for location in face["properties"]["bufferLocations"]["indices"]:
                    face_indices = index_array[location["startIndex"] : location["endIndex"]]
                    all_vertices.append(position_array[face_indices])
        else:
            # Unindexed geometry: buffer locations are float32 element offsets into position
            for face in faces:
                for location in face["properties"]["bufferLocations"]["indices"]:
                    start_vertex = location["startIndex"] // 3
                    end_vertex = location["endIndex"] // 3
                    all_vertices.append(position_array[start_vertex:end_vertex])

    return np.concatenate(all_vertices, axis=0)


def resolve_buffers(solid: dict, lod_level: Optional[int]) -> dict:
    """Resolve LOD to a plain buffer entry with path and sections.

    If the buffer uses LOD, selects the specified level (or the manifest default).
    Otherwise returns the buffer entry as-is.
    """
    buffers = solid["resources"]["buffers"]
    if buffers["type"] == "lod":
        resolved_level = lod_level if lod_level is not None else buffers.get("default", 0)
        return buffers["levels"][resolved_level]
    return buffers


def _find_section(sections: List[dict], name: str) -> Optional[dict]:
    """Find a named section within a buffer's section list."""
    return next((s for s in sections if s["name"] == name), None)
