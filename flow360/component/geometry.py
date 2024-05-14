"""
Geometry component
"""

from __future__ import annotations

import os
import re
from typing import List, Union

from ..cloud.rest_api import RestApi
from ..exceptions import Flow360FileError, Flow360ValueError
from ..log import log
from .interfaces import GeometryInterface
from .resource_base import Flow360Resource, Flow360ResourceBaseModel, ResourceDraft
from .utils import shared_account_confirm_proceed, validate_type

SUPPORTED_GEOMETRY_FILE_PATTERNS = [
    ".sat",
    ".sab",
    ".asat",
    ".asab",
    ".iam",
    ".catpart",
    ".catproduct",
    ".igs",
    ".iges",
    ".gt",
    ".prt",
    ".prt.*",
    ".asm.*",
    ".par",
    ".asm",
    ".psm",
    ".sldprt",
    ".sldasm",
    ".stp",
    ".step",
    ".x_t",
    ".xmt_txt",
    ".x_b",
    ".xmt_bin",
    ".3dm",
    ".ipt",
]


def _match_file_pattern(patterns, filename):
    for pattern in patterns:
        if re.search(pattern + "$", filename.lower()) is not None:
            return True
    return False


class Geometry(Flow360Resource):
    """
    Geometry component
    """

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=GeometryInterface,
            info_type_class=GeometryMeta,
            id=id,
        )
        self._params = None

    @classmethod
    def _from_meta(cls, meta: GeometryMeta):
        validate_type(meta, "meta", GeometryMeta)
        geometry = cls(id=meta.id)
        geometry._set_meta(meta)
        return geometry

    @property
    def info(self) -> GeometryMeta:
        return super().info

    @classmethod
    def _interface(cls):
        return GeometryInterface

    @classmethod
    def _meta_class(cls):
        """
        returns geometry meta info class: GeometryMeta
        """
        return GeometryMeta

    def _complete_upload(self, remote_file_names: List[str]):
        """
        Complete geometry files upload
        :return:
        """
        for remote_file_name in remote_file_names:
            resp = self.post({}, method=f"completeUpload?fileName={remote_file_name}")
            self._info = GeometryMeta(**resp)

    @classmethod
    def from_cloud(cls, geometry_id: str):
        """
        Get geometry from cloud
        :param geometry_id:
        :return:
        """
        return cls(geometry_id)

    @classmethod
    def from_file(
        cls,
        file_names: Union[List[str], str],
        name: str = None,
        tags: List[str] = None,
    ):
        """
        Create geometry from geometry files
        :param file_names:
        :param name:
        :param tags:
        :param solver_version:
        :return:
        """
        if isinstance(file_names, str):
            file_names = [file_names]
        return GeometryDraft(
            file_names=file_names,
            name=name,
            tags=tags,
        )


class GeometryMeta(Flow360ResourceBaseModel):
    """
    GeometryMeta component
    """

    def to_geometry(self) -> Geometry:
        """
        returns Geometry object from geometry meta info
        """
        return Geometry(self.id)


class GeometryDraft(ResourceDraft):
    """
    Geometry Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_names: List[str],
        name: str = None,
        tags: List[str] = None,
        solver_version=None,
    ):
        self._file_names = file_names
        self.name = name
        self.tags = tags
        self.solver_version = solver_version
        self._id = None
        self._validate()
        ResourceDraft.__init__(self)

    def _validate(self):
        self._validate_geometry()

    # pylint: disable=consider-using-f-string
    def _validate_geometry(self):
        if not isinstance(self.file_names, list):
            raise Flow360FileError("file_names field has to be a list.")
        for geometry_file in self.file_names:
            _, ext = os.path.splitext(geometry_file)
            if not _match_file_pattern(SUPPORTED_GEOMETRY_FILE_PATTERNS, geometry_file):
                raise Flow360FileError(
                    "Unsupported geometry file extensions: {}. Supported: [{}].".format(
                        ext.lower(), ", ".join(SUPPORTED_GEOMETRY_FILE_PATTERNS)
                    )
                )

            if not os.path.exists(geometry_file):
                raise Flow360FileError(f"{geometry_file} not found.")

        if self.name is None and len(self.file_names) > 1:
            raise Flow360ValueError(
                "name field is required if more than one geometry files are provided."
            )

    @property
    def file_names(self) -> List[str]:
        """geometry file"""
        return self._file_names

    # pylint: disable=protected-access
    # pylint: disable=duplicate-code
    def submit(self, progress_callback=None) -> Geometry:
        """submit geometry to cloud

        Parameters
        ----------
        progress_callback : callback, optional
            Use for custom progress bar, by default None

        Returns
        -------
        Geometry
            Geometry object with id
        """

        self._validate()
        name = self.name
        if name is None:
            name = os.path.splitext(os.path.basename(self.file_names[0]))[0]
        self.name = name

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        data = {
            "name": self.name,
            "tags": self.tags,
        }

        if self.solver_version:
            data["solverVersion"] = self.solver_version

        resp = RestApi(GeometryInterface.endpoint).post(data)
        info = GeometryMeta(**resp)
        self._id = info.id
        submitted_geometry = Geometry(self.id)

        remote_file_names = []
        for index, geometry_file in enumerate(self.file_names):
            _, ext = os.path.splitext(geometry_file)
            remote_file_name = f"geometry_{index}{ext}"
            file_name_to_upload = geometry_file
            submitted_geometry._upload_file(
                remote_file_name, file_name_to_upload, progress_callback=progress_callback
            )
            remote_file_names.append(remote_file_name)
        submitted_geometry._complete_upload(remote_file_names)
        log.info(f"Geometry successfully submitted: {submitted_geometry.short_description()}")
        return submitted_geometry
