"""
base component for examples
"""

import glob
import os
from abc import ABCMeta
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import requests

from flow360.log import log

from ..solver_version import Flow360Version
from ..utils import classproperty

here = os.path.dirname(os.path.abspath(__file__))


def download(url, filename):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    log.info(f"""The file ({os.path.basename(filename)}) is being downloaded, please wait.""")
    with open(filename, "wb") as fh:
        fh.write(response.content)


def version_parse(str):
    return Flow360Version(str.strip("lgte"))


def _as_list(value):
    """Wrap a scalar in a list; pass a list/tuple through unchanged (as a list)."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


#: asset slots whose local copy lives under a solver-version subfolder. Versioning
#: is a property of the slot, not the example, so it is resolved here once.
VERSIONED_SLOTS = frozenset({"mesh_json", "case_json", "case_yaml", "surface_json", "volume_json"})


class Asset:
    """One downloadable example slot: a single file or a set of files.

    Constructed with just the remote URL(s); the owning :class:`DownloadableAssets`
    records which slot it occupies and whether that slot is versioned. The object
    is stateless -- every path is derived from the owning example class on demand,
    so results stay correct across ``set_version()`` changes. The local path
    mirrors the URL's shape: a single URL yields a single path, a list of URLs
    yields a list of paths.

    The ``local://`` scheme marks a file bundled with the package rather than
    downloaded; such files already exist on disk, so :meth:`download` skips them.
    """

    def __init__(self, url):
        #: remote location(s) to download from: a URL string or list of strings
        self.url = url
        #: slot name this asset occupies (set by :class:`DownloadableAssets`)
        self.slot = None
        #: whether the local path includes the solver-version prefix
        self.versioned = False

    def local(self, owner):
        """Local path(s) under ``owner``, mirroring the URL's shape (str or list)."""
        prefix = owner._get_version_prefix() if self.versioned else ""
        paths = [owner._real_path(prefix, os.path.basename(u)) for u in _as_list(self.url)]
        return paths if isinstance(self.url, (list, tuple)) else paths[0]

    def is_downloaded(self, owner):
        """True only if every local file for this asset exists on disk."""
        return all(os.path.exists(p) for p in _as_list(self.local(owner)))

    def path(self, owner):
        """Local path(s); raise :class:`FileNotFoundError` if not yet downloaded."""
        if not self.is_downloaded(owner):
            raise FileNotFoundError("File not found. Run get_files() first to download files.")
        return self.local(owner)

    def download(self, owner):
        """Fetch any missing file(s); existing / ``local://`` files are skipped."""
        owner._get_file(self.url, self.local(owner))


@dataclass
class DownloadableAssets:
    """The files an example provides -- one optional field per slot.

    An example fills only the slots it ships; the rest stay ``None``. The fields
    here are the single definition of which slots exist, so editors can complete
    them and unknown slots are rejected at construction. ``extra`` is a free-form
    ``{key: url}`` dict for files addressed by key rather than a fixed slot.
    """

    geometry: Optional[Asset] = None
    mesh: Optional[Asset] = None
    mesh_json: Optional[Asset] = None
    case_json: Optional[Asset] = None
    case_yaml: Optional[Asset] = None
    surface_json: Optional[Asset] = None
    volume_json: Optional[Asset] = None
    extra: Optional[dict] = None

    def __post_init__(self):
        # Tell each filled Asset which slot it occupies and whether it is versioned.
        for f in fields(self):
            asset = getattr(self, f.name)
            if isinstance(asset, Asset):
                asset.slot = f.name
                asset.versioned = f.name in VERSIONED_SLOTS

    def filled(self):
        """Yield every filled ``Asset`` slot (``extra`` is handled separately)."""
        for f in fields(self):
            asset = getattr(self, f.name)
            if isinstance(asset, Asset):
                yield asset


class BaseTestCase(metaclass=ABCMeta):
    """Base class for bundled example cases.

    Subclasses set :attr:`name` and :attr:`downloadable_assets`; the attributes
    below expose the resolved, download-checked local paths.
    """

    _solver_version = None

    #: filled in by each subclass with the slots it provides
    downloadable_assets: DownloadableAssets = DownloadableAssets()

    #: each example's folder name / identifier; every subclass sets this
    name: str

    @classmethod
    def _get_version_prefix(cls):
        if cls._solver_version is None:
            return ""

        versionList = [
            os.path.basename(f) for f in glob.glob(os.path.join(here, cls.name, "release-*"))
        ]
        versionList.sort(key=version_parse)
        for v in versionList:
            if v == cls._solver_version:
                return v

            suffix = v[-2:]
            if suffix == "ge":
                if version_parse(v) >= version_parse(cls._solver_version):
                    return v
            elif suffix == "gt":
                if version_parse(v) > version_parse(cls._solver_version):
                    return v

        versionList.reverse()
        for v in versionList:
            suffix = v[-2:]
            if suffix == "le":
                if version_parse(v) <= version_parse(cls._solver_version):
                    return v
            elif suffix == "lt":
                if version_parse(v) < version_parse(cls._solver_version):
                    return v

        return ""

    @classmethod
    def _real_path(cls, *args):
        return os.path.join(here, cls.name, *args)

    @classmethod
    def _asset(cls, slot):
        asset = getattr(cls.downloadable_assets, slot)
        if asset is None:
            raise AttributeError(f"example {cls.__name__!r} provides no {slot!r} asset")
        return asset

    @classproperty
    def geometry(cls):
        return cls._asset("geometry").path(cls)

    @classproperty
    def mesh_filename(cls):
        return cls._asset("mesh").path(cls)

    @classproperty
    def mesh_json(cls):
        return cls._asset("mesh_json").path(cls)

    @classproperty
    def case_json(cls):
        return cls._asset("case_json").path(cls)

    @classproperty
    def case_yaml(cls):
        return cls._asset("case_yaml").path(cls)

    @classproperty
    def surface_json(cls):
        return cls._asset("surface_json").path(cls)

    @classproperty
    def volume_json(cls):
        return cls._asset("volume_json").path(cls)

    @classmethod
    def is_file_downloaded(cls, file):
        if not os.path.exists(file):
            raise FileNotFoundError("File not found. Run get_files() first to download files.")
        return file

    @classproperty
    def _extra(cls):
        versionPrefix = cls._get_version_prefix()
        return {
            key: cls._real_path(versionPrefix, os.path.basename(value))
            for key, value in cls.downloadable_assets.extra.items()
        }

    @classproperty
    def extra(cls):
        return {key: cls.is_file_downloaded(value) for key, value in cls._extra.items()}

    @classmethod
    def set_version(cls, version):
        cls._solver_version = version

    @classmethod
    def _get_file(cls, remote_filename, local_filename):
        for remote, local in zip(_as_list(remote_filename), _as_list(local_filename)):
            if not os.path.exists(local):
                download(remote, local)

    @classmethod
    def get_files(cls):
        for asset in cls.downloadable_assets.filled():
            asset.download(cls)
        if cls.downloadable_assets.extra:
            for key, value in cls.downloadable_assets.extra.items():
                cls._get_file(value, cls._extra[key])
