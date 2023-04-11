"""
base component for examples
"""


import os
import requests
import glob
from pathlib import Path
from abc import ABC, abstractmethod, abstractstaticmethod

from ..solver_version import Flow360Version

here = os.path.dirname(os.path.abspath(__file__))


def download(url, filename):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    with open(filename, "wb") as fh:
        fh.write(response.content)


def version_parse(str):
    return Flow360Version(str.strip("lgte"))


class url_base(ABC):
    @property
    @abstractmethod
    def geometry(self):
        pass

    @property
    @abstractmethod
    def mesh(self):
        pass

    @property
    @abstractmethod
    def mesh_json(self):
        pass

    @property
    @abstractmethod
    def case_json(self):
        pass

    @property
    @abstractmethod
    def case_yaml(self):
        pass

    @abstractmethod
    def surface_json(self):
        pass

    @abstractmethod
    def surface_yaml(self):
        pass


class BaseTestCase(ABC):
    _solver_version = None

    @property
    @abstractstaticmethod
    def name():
        pass

    @property
    @abstractstaticmethod
    def url(cls) -> url_base:
        pass

    @classmethod
    def _get_version_prefix(cls):
        if cls._solver_version == None:
            return ""

        versionList = [
            os.path.basename(f) for f in glob.glob(os.path.join(here, cls.name, "release-*"))
        ]
        versionList.sort(key=version_parse)

        for v in versionList:
            try:
                if version_parse(v) == version_parse(cls._solver_version):
                    return v
            except:
                pass

            suffix = v[-2:]
            if suffix == "ge":
                if version_parse(v) >= version_parse(cls._solver_version):
                    return v
            elif suffix == "gt":
                if version_parse(v) > version_parse(cls._solver_version):
                    return v

        versionList.reverse()
        for v in versionList:
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
    @property
    def _geometry_filename(cls):
        return cls._real_path(os.path.basename(cls.url.geometry))

    @classmethod
    @property
    def _mesh_filename(cls):
        return cls._real_path(os.path.basename(cls.url.mesh))

    @classmethod
    @property
    def _mesh_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.mesh_json))

    @classmethod
    @property
    def _case_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.case_json))

    @classmethod
    @property
    def _case_yaml(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.case_yaml))

    @classmethod
    @property
    def _surface_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.surface_json))

    @classmethod
    @property
    def _volume_json(cls):
        versionPrefix = cls._get_version_prefix()
        return cls._real_path(versionPrefix, os.path.basename(cls.url.volume_json))

    @classmethod
    def is_file_downloaded(cls, file):
        if not os.path.exists(file):
            raise RuntimeError(f"File not found. Run get_files() first to download files.")
        return file

    @classmethod
    @property
    def geometry(cls):
        return cls.is_file_downloaded(cls._geometry_filename)

    @classmethod
    @property
    def mesh_filename(cls):
        return cls.is_file_downloaded(cls._mesh_filename)

    @classmethod
    @property
    def mesh_json(cls):
        return cls.is_file_downloaded(cls._mesh_json)

    @classmethod
    @property
    def case_json(cls):
        return cls.is_file_downloaded(cls._case_json)

    @classmethod
    @property
    def case_yaml(cls):
        return cls.is_file_downloaded(cls._case_yaml)

    @classmethod
    @property
    def surface_json(cls):
        return cls.is_file_downloaded(cls._surface_json)

    @classmethod
    @property
    def volume_json(cls):
        return cls.is_file_downloaded(cls._volume_json)

    @classmethod
    def set_version(cls, version):
        cls._solver_version = version

    @classmethod
    def get_files(cls):
        try:
            if not os.path.exists(cls._mesh_filename):
                download(cls.url.mesh, cls._mesh_filename)
        except AttributeError:
            pass

        try:
            if not os.path.exists(cls._mesh_json):
                download(cls.url.mesh_json, cls._mesh_json)
        except AttributeError:
            pass

        try:
            if not os.path.exists(cls._case_json):
                download(cls.url.case_json, cls._case_json)
        except AttributeError:
            pass

        try:
            if not os.path.exists(cls._case_yaml):
                download(cls.url.case_yaml, cls._case_yaml)
        except AttributeError:
            pass
