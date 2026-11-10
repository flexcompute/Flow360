"""General-purpose size-based LRU disk cache for cloud file downloads.

Stores files under ``~/.flow360/cache/<namespace>/<resource_id>/<file_path>``
with a configurable total size limit.  Eviction granularity is the resource
directory — all files for a resource are deleted together to avoid partial
state (e.g. manifest present but bin evicted).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from ..log import log

CLOUD_FILE_CACHE_MAX_SIZE_MB: int = 2048  # default 2 GB, user-adjustable

_shared_cache_instance: Optional["CloudFileCache"] = None


def get_shared_cloud_file_cache() -> "CloudFileCache":
    """Return the module-level shared CloudFileCache instance (created on first call)."""
    global _shared_cache_instance  # pylint: disable=global-statement
    if _shared_cache_instance is None:
        _shared_cache_instance = CloudFileCache()
    return _shared_cache_instance


class CloudFileCache:
    """Size-based LRU disk cache.

    Keys are ``(namespace, resource_id, file_path)`` triples.
    All namespaces share a single total-size budget.
    """

    def __init__(
        self,
        cache_root: Optional[Path] = None,
        max_size_bytes: Optional[int] = None,
    ) -> None:
        self._cache_root = (cache_root or Path("~/.flow360/cache")).expanduser()
        self._max_size_bytes = (
            CLOUD_FILE_CACHE_MAX_SIZE_MB * 1024 * 1024 if max_size_bytes is None else max_size_bytes
        )
        self._disabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, namespace: str, resource_id: str, file_path: str) -> Optional[bytes]:
        """Return cached bytes or ``None``.  Touches ``.last_access`` on hit."""
        if self._disabled:
            return None

        target = self._file_path(namespace, resource_id, file_path)
        if not target.is_file():
            return None

        try:
            data = target.read_bytes()
        except OSError:
            return None

        self._touch_last_access(namespace, resource_id)
        return data

    def put(self, namespace: str, resource_id: str, file_path: str, data: bytes) -> None:
        """Write *data* to disk, evicting oldest resources if over size limit."""
        if self._disabled:
            return

        # Skip caching entries that exceed the entire cache budget
        if len(data) > self._max_size_bytes:
            return

        try:
            # Account for the file being overwritten (net size delta, not gross)
            target = self._file_path(namespace, resource_id, file_path)
            existing_size = target.stat().st_size if target.is_file() else 0
            net_incoming = len(data) - existing_size

            current_resource_dir = self._resource_dir(namespace, resource_id)
            self._evict_if_needed(net_incoming, protect=current_resource_dir)

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            self._touch_last_access(namespace, resource_id)
        except OSError as exc:
            log.warning(f"CloudFileCache: disk write failed ({exc}), disabling cache")
            self._disabled = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _file_path(self, namespace: str, resource_id: str, file_path: str) -> Path:
        target = (self._cache_root / namespace / resource_id / file_path).resolve()
        if not target.is_relative_to(self._cache_root.resolve()):
            raise ValueError(f"Path traversal detected in cache key: {file_path!r}")
        return target

    def _resource_dir(self, namespace: str, resource_id: str) -> Path:
        return self._cache_root / namespace / resource_id

    def _last_access_path(self, namespace: str, resource_id: str) -> Path:
        return self._resource_dir(namespace, resource_id) / ".last_access"

    def _touch_last_access(self, namespace: str, resource_id: str) -> None:
        """Create or update the ``.last_access`` sentinel in the resource dir."""
        path = self._last_access_path(namespace, resource_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _collect_resource_dirs(self) -> Tuple[int, List[Tuple[float, int, Path]]]:
        """Scan cache and return ``(total_size, [(mtime, size, dir), ...])``.

        Single pass: computes both the aggregate size and per-resource metadata
        needed for LRU eviction.
        """
        entries: List[Tuple[float, int, Path]] = []
        total_size = 0
        if not self._cache_root.exists():
            return total_size, entries

        for namespace_dir in self._cache_root.iterdir():
            if not namespace_dir.is_dir():
                continue
            for resource_dir in namespace_dir.iterdir():
                if not resource_dir.is_dir():
                    continue
                last_access = resource_dir / ".last_access"
                mtime = last_access.stat().st_mtime if last_access.exists() else 0.0
                size = sum(
                    f.stat().st_size
                    for f in resource_dir.rglob("*")
                    if f.is_file() and f.name != ".last_access"
                )
                total_size += size
                entries.append((mtime, size, resource_dir))
        return total_size, entries

    def _evict_if_needed(self, incoming_bytes: int, protect: Optional[Path] = None) -> None:
        """Delete oldest resource dirs until total size + *incoming_bytes* fits the budget.

        *protect*, if given, is a resource directory that must not be evicted
        (the resource currently being populated by the caller).
        """
        current_size, entries = self._collect_resource_dirs()
        if current_size + incoming_bytes <= self._max_size_bytes:
            return

        # Sort by last-access time ascending (oldest first)
        entries.sort(key=lambda e: e[0])

        for _mtime, size, resource_dir in entries:
            if current_size + incoming_bytes <= self._max_size_bytes:
                break
            if protect is not None and resource_dir == protect:
                continue
            shutil.rmtree(resource_dir, ignore_errors=True)
            current_size -= size
