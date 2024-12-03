"""
base component for examples
"""

import glob
import os
from abc import ABCMeta, abstractmethod, abstractstaticmethod
from pathlib import Path

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


class url_base(metaclass=ABCMeta):
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

    @property
    @abstractmethod
    def extra(self):
        """extra"""


class BaseTestCase(metaclass=ABCMeta):
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

    @classproperty
    def _extra(cls):
        versionPrefix = cls._get_version_prefix()
        return {
            key: cls._real_path(versionPrefix, os.path.basename(value))
            for key, value in cls.url.extra.items()
        }

    @classmethod
    def is_file_downloaded(cls, file):
        if not os.path.exists(file):
            raise FileNotFoundError("File not found. Run get_files() first to download files.")
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

    @classproperty
    def extra(cls):
        return {key: cls.is_file_downloaded(value) for key, value in cls._extra.items()}

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
        if hasattr(cls.url, "geometry"):
            cls._get_file(cls.url.geometry, cls._geometry_filename)
        if hasattr(cls.url, "surface_json"):
            cls._get_file(cls.url.surface_json, cls._surface_json)
