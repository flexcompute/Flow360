"""Tests for AssetBase.download segment-aware pattern matching."""

import os
from types import SimpleNamespace

import pytest

from flow360.component.simulation.web.asset_base import AssetBase
from flow360.exceptions import Flow360FileError, Flow360ValueError


class _StubWebApi:
    info = SimpleNamespace(name="dummy-asset")

    # pylint: disable=unused-argument
    def _download_file(self, file_name, to_folder=".", overwrite=True):
        dest = os.path.join(to_folder, os.path.basename(file_name))
        with open(dest, "w", encoding="utf-8"):
            pass
        return dest


class _NoneReturnWebApi(_StubWebApi):
    # Mimics _local_download_overwrite (from_local_storage): writes file, returns None.
    def _download_file(self, file_name, to_folder=".", overwrite=True):
        super()._download_file(file_name, to_folder=to_folder, overwrite=overwrite)
        return None


class _DummyAsset(AssetBase):
    _cloud_resource_type_name = "Geometry"

    # pylint: disable=super-init-not-called
    def __init__(self, files):
        self.id = "dummy"
        self._webapi = _StubWebApi()
        self._files = files

    def get_download_file_list(self):
        return [{"fileName": name} for name in self._files]

    def get_dynamic_default_settings(self, simulation_dict):
        pass


def test_star_does_not_cross_separator(tmp_path):
    asset = _DummyAsset(["a.csv", "results/b.csv"])
    paths = asset.download("*.csv", to_folder=str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["a.csv"]


def test_folder_scoped_pattern(tmp_path):
    asset = _DummyAsset(["a.csv", "results/b.csv", "logs/c.csv"])
    paths = asset.download("results/*.csv", to_folder=str(tmp_path))
    assert [os.path.relpath(p, tmp_path) for p in paths] == [os.path.join("results", "b.csv")]


def test_doublestar_matches_any_depth(tmp_path):
    # '**' = zero or more dirs, so it also matches root-level files.
    asset = _DummyAsset(["a.csv", "results/b.csv", "deep/sub/c.csv"])
    paths = asset.download("**/*.csv", to_folder=str(tmp_path))
    rel = sorted(os.path.relpath(p, tmp_path) for p in paths)
    assert rel == [
        "a.csv",
        os.path.join("deep", "sub", "c.csv"),
        os.path.join("results", "b.csv"),
    ]


def test_globstar_matches_zero_or_more_dirs(tmp_path):
    # results/**/*.csv must match both results/foo.csv (zero dirs) and nested files.
    asset = _DummyAsset(["results/foo.csv", "results/sub/bar.csv", "other/x.csv"])
    paths = asset.download("results/**/*.csv", to_folder=str(tmp_path))
    rel = sorted(os.path.relpath(p, tmp_path) for p in paths)
    assert rel == [
        os.path.join("results", "foo.csv"),
        os.path.join("results", "sub", "bar.csv"),
    ]


def test_download_returns_path_when_download_file_returns_none(tmp_path):
    # from_local_storage assets copy the file but return None; download() must not crash.
    asset = _DummyAsset(["mesh.cgns"])
    asset._webapi = _NoneReturnWebApi()
    paths = asset.download("*.cgns", to_folder=str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["mesh.cgns"]
    assert all(os.path.isabs(p) and os.path.isfile(p) for p in paths)


def test_download_mirrors_cloud_subfolders(tmp_path):
    # Same base name in different cloud folders must not collide.
    asset = _DummyAsset(["results/x.csv", "logs/x.csv"])
    paths = asset.download("**/*.csv", to_folder=str(tmp_path))
    rel = sorted(os.path.relpath(p, tmp_path) for p in paths)
    assert rel == [os.path.join("logs", "x.csv"), os.path.join("results", "x.csv")]
    assert all(os.path.isfile(p) for p in paths)


def test_download_is_case_insensitive(tmp_path):
    asset = _DummyAsset(["Part.CATPart", "mesh.STL"])
    paths = asset.download(["*.catpart", "*.stl"], to_folder=str(tmp_path))
    assert sorted(os.path.basename(p) for p in paths) == ["Part.CATPart", "mesh.STL"]


def test_download_raises_when_nothing_matches(tmp_path):
    asset = _DummyAsset(["a.csm"])
    with pytest.raises(Flow360FileError, match=r"\*\.xyz"):
        asset.download("*.xyz", to_folder=str(tmp_path))


def test_download_partial_match_downloads_matches(tmp_path):
    # One pattern matches, one does not: matched files still download, no raise.
    asset = _DummyAsset(["a.csm"])
    paths = asset.download(["*.csm", "*.xyz"], to_folder=str(tmp_path))
    assert sorted(os.path.basename(p) for p in paths) == ["a.csm"]


def test_default_download_is_root_only(tmp_path):
    # Default '*.ext' patterns are segment-aware -> skip results/logs subfolders.
    class _WithDefaults(_DummyAsset):
        _default_download_patterns = ["*.cgns"]

    asset = _WithDefaults(["mesh.cgns", "results/out.cgns", "logs/run.cgns"])
    paths = asset.download(to_folder=str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["mesh.cgns"]


def test_download_raises_without_patterns_or_defaults(tmp_path):
    asset = _DummyAsset(["a.csm"])  # inherits _default_download_patterns = None
    with pytest.raises(Flow360ValueError):
        asset.download(to_folder=str(tmp_path))


def test_real_resources_have_default_input_patterns():
    # pylint: disable=import-outside-toplevel
    from flow360.component.geometry import Geometry
    from flow360.component.surface_mesh_v2 import SurfaceMeshV2
    from flow360.component.volume_mesh import VolumeMeshV2

    # Geometry default is the union of CAD source files and surface-mesh files.
    assert {"*.csm", "*.cgns", "*.stl", "*.mapbc"} <= set(Geometry._default_download_patterns)
    # Surface mesh accepts stl; volume mesh does not.
    assert "*.stl" in SurfaceMeshV2._default_download_patterns
    assert "*.cgns" in VolumeMeshV2._default_download_patterns
    assert "*.stl" not in VolumeMeshV2._default_download_patterns
