"""Fetch Flow360 *benchmark* case assets from the public benchmark bucket.

Everything a benchmark case publishes lives under one per-case prefix, so the
per-case reproducer notebooks can fetch what they need with no credentials and no
hard-coded URLs::

    benchmarks/<case_id>/root_assets/index.json          # ["<project_id>", ...]
    benchmarks/<case_id>/root_assets/<project_id>/snapshot.json   # {"files": [...], ...}
    benchmarks/<case_id>/root_assets/<project_id>/<files>
    benchmarks/<case_id>/<kind>/manifest.json            # ["<relpath>", ...]
    benchmarks/<case_id>/<kind>/<relpath>

- **Root assets** (``kind="root_assets"``) — a project's geometry or volume mesh.
  A single-project case resolves its one project via ``index.json``; a
  multi-project case (several mesh levels / pitches) passes the specific
  ``project_id``.
- **Case input data** (any other ``kind`` — ``"ref_data"``, ``"turb"``,
  ``"BET_JSONS"``, …) — the local files a case's ``run.py`` / ``postprocess.py``
  read, recreated under ``<kind>/…``.

``download_benchmark_assets`` downloads the published files over HTTPS and
returns their local paths.

This is intentionally *not* a general "download any cloud project's assets": a
project you do not own cannot be read through the SDK. It only serves cases the
Flow360 benchmark pipeline has published to the public bucket.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import List

BENCHMARK_ASSET_BASE = "https://simcloud-public-1.s3.amazonaws.com/benchmarks"

#: ``kind`` value that selects a case's snapshotted root asset (geometry / volume
#: mesh) rather than one of its input-data folders.
ROOT_ASSETS = "root_assets"


def _asset_url(*segments: str, relpath: str | None = None) -> str:
    """Object URL under :data:`BENCHMARK_ASSET_BASE`, percent-encoded.

    ``segments`` are single path components (case id, kind, project id, file
    name) and are encoded whole; ``relpath`` is a manifest-relative path whose
    ``/`` separators are kept, so names containing spaces, ``#`` or ``?`` still
    address the right object.
    """
    parts = [urllib.parse.quote(segment, safe="") for segment in segments]
    if relpath is not None:
        parts.append(urllib.parse.quote(relpath))
    return "/".join([BENCHMARK_ASSET_BASE, *parts])


def _read_json(url: str, *, what: str) -> object:
    try:
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            raise FileNotFoundError(
                f"{what} not found in the public benchmark bucket ({url}). "
                "download_benchmark_assets only serves assets published by the "
                "Flow360 benchmark pipeline."
            ) from exc
        raise


def _resolve_root_asset_project(case_id: str, project_id: str | None) -> str:
    """The project id whose root-asset snapshot to fetch. When ``project_id`` is
    given (multi-project cases) it is used directly; otherwise the case's
    ``index.json`` must list exactly one project."""
    if project_id is not None:
        return project_id
    url = _asset_url(case_id, ROOT_ASSETS, "index.json")
    index = _read_json(url, what=f"root-asset index for case {case_id!r}")
    project_ids = index if isinstance(index, list) else []
    if len(project_ids) == 1:
        return project_ids[0]
    if not project_ids:
        # An empty or malformed index means the case published no root asset at
        # all — the same "nothing there" condition as a missing index, so there
        # is no project for the caller to pick with project_id=.
        raise FileNotFoundError(
            f"Root-asset index for case {case_id!r} lists no projects ({url}). "
            "download_benchmark_assets only serves assets published by the "
            "Flow360 benchmark pipeline."
        )
    raise ValueError(
        f"Case {case_id!r} has {len(project_ids)} root-asset projects "
        f"{project_ids!r}; pass project_id= to pick one."
    )


def download_benchmark_assets(
    case_id: str,
    kind: str,
    project_id: str | None = None,
    to_folder: str = ".",
) -> List[str]:
    """Download a benchmark case's published assets.

    Parameters
    ----------
    case_id : str
        Benchmark case id, e.g. ``"NLF_airfoil"`` — everything for the case lives
        under ``benchmarks/<case_id>/``.
    kind : str
        ``"root_assets"`` for the case's snapshotted root asset (geometry or
        volume mesh), or the name of an input-data folder (e.g. ``"ref_data"``,
        ``"turb"``, ``"BET_JSONS"``).
    project_id : str, optional
        Only for ``kind="root_assets"`` on a **multi-project** case (several mesh
        levels / pitches): which project's root asset to fetch. Omit for a
        single-project case — it is resolved from the case's ``index.json``.
    to_folder : str
        Local destination folder (created if missing). Defaults to the current
        directory. Root-asset files land directly under ``to_folder``; input-data
        files are recreated under ``to_folder/<kind>/…`` so the scripts read their
        original relative paths unchanged.

    Returns
    -------
    List[str]
        Absolute paths of the downloaded files. For a volume-mesh root asset pass
        the first entry to :func:`Project.from_volume_mesh`; for a geometry root
        asset pass the whole list to :func:`Project.from_geometry`.

    Raises
    ------
    FileNotFoundError
        If nothing is published for the case/kind. This helper only serves assets
        published by the Flow360 benchmark pipeline; it is not a general-purpose
        cloud-project download.
    ValueError
        If ``kind="root_assets"`` on a multi-project case and ``project_id`` is
        not given.
    """
    if kind == ROOT_ASSETS:
        pid = _resolve_root_asset_project(case_id, project_id)
        prefix = (case_id, ROOT_ASSETS, pid)
        manifest = _read_json(
            _asset_url(*prefix, "snapshot.json"),
            what=f"root-asset snapshot for {case_id}/{pid}",
        )
        raw_files = manifest.get("files") if isinstance(manifest, dict) else None
        files = raw_files if isinstance(raw_files, list) else None
        dest_prefix = ""  # root-asset files land flat under to_folder
    else:
        prefix = (case_id, kind)
        manifest = _read_json(
            _asset_url(*prefix, "manifest.json"), what=f"{kind} manifest for case {case_id!r}"
        )
        files = manifest if isinstance(manifest, list) else None
        dest_prefix = kind  # recreate ./<kind>/<relpath> so scripts find them

    files = files or []
    if not files:
        raise FileNotFoundError(
            f"Benchmark manifest for case {case_id!r}, kind {kind!r} lists no files."
        )

    paths: List[str] = []
    # Manifest entries are untrusted input from the published S3 store. Resolve
    # each under the destination root and reject anything that escapes it (an
    # absolute path makes os.path.join drop the prefix; ".." segments climb out),
    # so a bad or compromised publish cannot write outside to_folder.
    root = os.path.realpath(os.path.join(to_folder, dest_prefix))
    for name in files:
        dest = os.path.realpath(os.path.join(root, name))
        if os.path.isabs(name) or (dest != root and not dest.startswith(root + os.sep)):
            raise ValueError(f"Unsafe benchmark asset path in manifest: {name!r}")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        urllib.request.urlretrieve(_asset_url(*prefix, relpath=name), dest)
        paths.append(dest)
    return paths
