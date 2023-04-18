"""
base component for examples
"""


import glob
import os
from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path

import requests

from ..solver_version import Flow360Version

here = os.path.dirname(os.path.abspath(__file__))


def download(url, filename):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    with open(filename, "wb") as fh:
        fh.write(response.content)


def version_parse(str):
    return Flow360Version(str.strip("lgte"))


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class url_base(ABC):
    @property
    @abstractmethod
    def geometry(self):
        """geometry"""

    @property
    @abstractmethod
    def mesh(self):
        """mesh"""

    @property
    @abstractmethod
    def mesh_json(self):
        """mesh_json"""

    @property
    @abstractmethod
    def case_json(self):
        """case_json"""

    @property
    @abstractmethod
    def case_yaml(self):
        """case_yaml"""

    @property
    @abstractmethod
    def surface_json(self):
        """surface_json"""

    @property
    @abstractmethod
    def surface_yaml(self):
        """surface_yaml"""


class BaseTestCase(ABC):
    _solver_version = None

    @property
    @abstractstaticmethod
    def name(cls):
        """name"""

    @property
    @abstractstaticmethod
    def url(cls) -> url_base:
        """url"""

    @classmethod
    def _get_version_prefix(cls):
        if cls._solver_version == None:
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

    @classproperty
    def _geometry_filename(cls):
        return cls._real_path(os.path.basename(cls.url.geometry))

    @classproperty
    def _mesh_filename(cls):
        return cls._real_path(os.path.basename(cls.url.mesh))

    @classproperty
    def _mesh_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.mesh_json))

    @classproperty
    def _case_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.case_json))

    @classproperty
    def _case_yaml(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.case_yaml))

    @classproperty
    def _surface_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.surface_json))

    @classproperty
    def _volume_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.volume_json))

    @classmethod
    def is_file_downloaded(cls, file):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found. Run get_files() first to download files.")
        return file

    @classproperty
    def geometry(cls):
        return cls.is_file_downloaded(cls._geometry_filename)

    @classproperty
    def mesh_filename(cls):
        return cls.is_file_downloaded(cls._mesh_filename)

    @classproperty
    def mesh_json(cls):
        return cls.is_file_downloaded(cls._mesh_json)

    @classproperty
    def case_json(cls):
        return cls.is_file_downloaded(cls._case_json)

    @classproperty
    def case_yaml(cls):
        return cls.is_file_downloaded(cls._case_yaml)

    @classproperty
    def surface_json(cls):
        return cls.is_file_downloaded(cls._surface_json)

    @classproperty
    def volume_json(cls):
        return cls.is_file_downloaded(cls._volume_json)

    @classmethod
    def set_version(cls, version):
        cls._solver_version = version

    @classmethod
    def _get_file(cls, remote_filename, local_filename):
        if not os.path.exists(local_filename):
            download(remote_filename, local_filename)

    @classmethod
    def get_files(cls):
        if hasattr(cls.url, "mesh"):
            cls._get_file(cls.url.mesh, cls._mesh_filename)
        if hasattr(cls.url, "mesh_json"):
            cls._get_file(cls.url.mesh_json, cls._mesh_json)
        if hasattr(cls.url, "case_json"):
            cls._get_file(cls.url.case_json, cls._case_json)
        if hasattr(cls.url, "case_yaml"):
            cls._get_file(cls.url.case_yaml, cls._case_yaml)
