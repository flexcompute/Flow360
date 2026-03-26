"""Download and cache tessellation files (manifest + bin) for OBB computation.

Two-layer cache strategy:
  L1: in-memory dict (instance-level, lives with the draft object)
  L2: disk cache via CloudFileCache (cross-session persistence)
"""

from __future__ import annotations

import os
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

import numpy as np

from flow360.cloud.file_cache import CloudFileCache
from flow360.component.resource_base import Flow360Resource
from flow360.log import log

from .uvf_parser import extract_face_vertices, parse_manifest, resolve_buffers


class TessellationFileLoader:  # pylint: disable=too-few-public-methods
    """Downloads and caches tessellation files (manifest + bin) for OBB computation.

    L1: in-memory cache (instance-level, lives with the draft object)
    L2: disk cache via CloudFileCache (cross-session persistence)
    """

    NAMESPACE = "tessellation"
    MANIFEST_PATH = "visualize/manifest/manifest.json"
    MANIFEST_DIR = "visualize/manifest"

    def __init__(
        self,
        geometry_resources: Dict[str, Flow360Resource],
        cloud_cache: CloudFileCache,
    ) -> None:
        """
        Args:
            geometry_resources: {geometry_id: Flow360Resource} for all active geometries.
            cloud_cache: Shared disk cache instance.
        """
        self._resources = geometry_resources
        self._cloud_cache = cloud_cache

        # L1 memory caches
        self._manifest_cache: Dict[str, List[dict]] = {}
        self._memory_cache: Dict[Tuple[str, str], bytes] = {}

        # Lazy-built indices
        self._face_to_geometry: Optional[Dict[str, str]] = None

    def load_vertices(self, face_ids: List[str], lod_level: Optional[int] = None) -> np.ndarray:
        """Extract vertices for the given face IDs across all geometries.

        This is the main entry point for DraftContext.compute_obb().
        Handles manifest loading, face-to-geometry mapping, bin downloading,
        and vertex extraction.

        Args:
            face_ids: Face IDs to extract vertices for.
            lod_level: LOD level override. None means use the manifest default.

        Returns:
            (N, 3) float32 array of vertex positions.
        """
        self._ensure_manifests_loaded()
        self._ensure_face_index_built()

        # Group face_ids by geometry
        geometry_faces: Dict[str, List[str]] = {}
        for fid in face_ids:
            geometry_id = self._face_to_geometry[fid]
            geometry_faces.setdefault(geometry_id, []).append(fid)

        all_vertices: List[np.ndarray] = []
        for geometry_id, geo_face_ids in geometry_faces.items():
            manifest = self._manifest_cache[geometry_id]
            bin_data_map = self._load_required_bins(geometry_id, manifest, geo_face_ids, lod_level)
            vertices = extract_face_vertices(manifest, geo_face_ids, bin_data_map, lod_level)
            all_vertices.append(vertices)

        return np.concatenate(all_vertices, axis=0)

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def _ensure_manifests_loaded(self) -> None:
        """Lazy-load manifests for all geometries (L1 -> L2 -> download)."""
        missing = [gid for gid in self._resources if gid not in self._manifest_cache]
        if missing:
            log.info("OBB: loading tessellation data from cloud...")
        for geometry_id in self._resources:
            if geometry_id in self._manifest_cache:
                continue
            raw = self._load_file(geometry_id, self.MANIFEST_PATH)
            self._manifest_cache[geometry_id] = parse_manifest(raw)

    # ------------------------------------------------------------------
    # Face index
    # ------------------------------------------------------------------

    def _ensure_face_index_built(self) -> None:
        """Build global face_id -> geometry_id index from cached manifests."""
        if self._face_to_geometry is not None:
            return
        self._face_to_geometry = {}
        for geometry_id, manifest in self._manifest_cache.items():
            for entry in manifest:
                entry_id = entry.get("id")
                if entry_id is not None:
                    self._face_to_geometry[entry_id] = geometry_id

    # ------------------------------------------------------------------
    # Bin file resolution
    # ------------------------------------------------------------------

    def _load_required_bins(  # pylint: disable=too-many-locals
        self,
        geometry_id: str,
        manifest: List[dict],
        face_ids: List[str],
        lod_level: Optional[int],
    ) -> Dict[str, bytes]:
        """Determine which bin files are needed and load them.

        Only downloads bins that contain data for the requested face_ids.
        """
        by_id = {entry["id"]: entry for entry in manifest}

        needed_bin_paths: set = set()
        for fid in face_ids:
            face = by_id[fid]
            parent_id = face["attributions"]["packedParentId"]
            solid = by_id[parent_id]
            buffer_entry = resolve_buffers(solid, lod_level)
            needed_bin_paths.add(buffer_entry["path"])

        bin_data_map: Dict[str, bytes] = {}
        for bin_path in needed_bin_paths:
            full_path = f"{self.MANIFEST_DIR}/{bin_path}"
            raw = self._load_file(geometry_id, full_path)
            bin_data_map[bin_path] = raw

        return bin_data_map

    # ------------------------------------------------------------------
    # Two-layer file loading (L1 -> L2 -> download)
    # ------------------------------------------------------------------

    def _load_file(self, geometry_id: str, file_path: str) -> bytes:
        """Load a file through L1 memory -> L2 disk -> cloud download."""
        key = (geometry_id, file_path)

        # L1: memory
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached

        # L2: disk
        data = self._cloud_cache.get(self.NAMESPACE, geometry_id, file_path)
        if data is not None:
            self._memory_cache[key] = data
            return data

        # Download from cloud
        log.debug(f"OBB: downloading tessellation file '{file_path}' ({geometry_id})")
        data = self._download_to_bytes(geometry_id, file_path)
        self._memory_cache[key] = data
        self._cloud_cache.put(self.NAMESPACE, geometry_id, file_path, data)
        return data

    def _download_to_bytes(self, geometry_id: str, file_path: str) -> bytes:
        """Download a file from cloud storage and return raw bytes."""
        resource = self._resources[geometry_id]
        # Use the remote file's extension as suffix so _download_file doesn't
        # append an extra extension to the temp path (see s3_utils.get_local_filename_and_create_folders).
        _, ext = os.path.splitext(file_path)
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
        try:
            # pylint: disable=protected-access
            resource._download_file(file_path, to_file=tmp_path, log_error=False, verbose=False)
            with open(tmp_path, "rb") as fh:
                return fh.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
