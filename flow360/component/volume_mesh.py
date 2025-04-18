"""
Volume mesh component
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import os.path
import threading
from enum import Enum
from functools import cached_property
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import pydantic as pd_v2

# This will be split into two files in a future commit...
# For now we keep it to not mix new features with file
# structure refactors
import pydantic.v1 as pd

from flow360.cloud.compress_upload import compress_and_upload_chunks
from flow360.cloud.flow360_requests import (
    CopyExampleVolumeMeshRequest,
    LengthUnitType,
    NewVolumeMeshRequestV2,
)
from flow360.cloud.heartbeat import post_upload_heartbeat
from flow360.cloud.rest_api import RestApi
from flow360.component.utils import VolumeMeshFile
from flow360.component.v1.cloud.flow360_requests import NewVolumeMeshRequest
from flow360.component.v1.meshing.params import VolumeMeshingParams
from flow360.exceptions import (
    Flow360CloudFileError,
    Flow360FileError,
    Flow360NotImplementedError,
    Flow360RuntimeError,
    Flow360ValueError,
)
from flow360.flags import Flags
from flow360.log import log
from flow360.solver_version import Flow360Version

from .case import Case, CaseDraft
from .interfaces import VolumeMeshInterface, VolumeMeshInterfaceV2
from .resource_base import (
    AssetMetaBaseModel,
    AssetMetaBaseModelV2,
    Flow360Resource,
    Flow360ResourceListBase,
    ResourceDraft,
)
from .results.base_results import PerEntityResultCSVModel
from .simulation.entity_info import VolumeMeshEntityInfo
from .simulation.primitives import GenericVolume, Surface
from .simulation.web.asset_base import AssetBase
from .types import COMMENTS
from .utils import (
    CompressionFormat,
    MeshFileFormat,
    MeshNameParser,
    UGRIDEndianness,
    shared_account_confirm_proceed,
    validate_type,
    zstd_compress,
)
from .v1.boundaries import NoSlipWall
from .v1.flow360_params import Flow360MeshParams, Flow360Params, _GenericBoundaryWrapper
from .v1.params_base import params_generic_validator
from .validator import Validator

try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def get_datatype(dataset):
    """
    Get datatype of dataset
    :param dataset:
    :return:
    """
    data_raw = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(data_raw)
    data_str = "".join([chr(i) for i in dataset])
    return data_str


def get_no_slip_walls(params: Union[Flow360Params, Flow360MeshParams]):
    """
    Get wall boundary names
    :param params:
    :return:
    """
    assert params

    if (
        isinstance(params, Flow360MeshParams)
        and params.boundaries
        and params.boundaries.no_slip_walls
    ):
        return params.boundaries.no_slip_walls

    if isinstance(params, Flow360Params) and params.boundaries:
        return [
            wall_name
            for wall_name, wall in params.boundaries.dict().items()
            # pylint: disable=no-member
            if wall_name != COMMENTS and _GenericBoundaryWrapper(v=wall).v.type == NoSlipWall().type
        ]

    return []


def get_boundaries_from_sliding_interfaces(params: Union[Flow360Params, Flow360MeshParams]):
    """
    Get wall boundary names
    :param params:
    :return:
    """
    assert params
    res = []

    # Sliding interfaces are deprecated - we need to handle this somehow
    # if params.sliding_interfaces and params.sliding_interfaces.rotating_patches:
    #    res += params.sliding_interfaces.rotating_patches[:]
    # if params.sliding_interfaces and params.sliding_interfaces.stationary_patches:
    #    res += params.sliding_interfaces.stationary_patches[:]
    return res


# pylint: disable=too-many-branches
def get_boundaries_from_file(cgns_file: str, solver_version: str = None):
    """
    Get boundary names from CGNS file
    :param cgns_file:
    :param solver_version:
    :return:
    """
    names = []
    with h5py.File(cgns_file, "r") as h5_file:
        base = h5_file["Base"]
        for zone_name, zone in base.items():
            if zone_name == " data":
                continue
            if zone.attrs["label"].decode() != "Zone_t":
                continue
            zone_type = get_datatype(base[f"{zone_name}/ZoneType/ data"])
            if zone_type not in ["Structured", "Unstructured"]:
                continue
            for section_name, section in zone.items():
                if section_name == " data":
                    continue
                if "label" not in section.attrs:
                    continue
                if solver_version and Flow360Version(solver_version) < Flow360Version(
                    "release-22.2.1.0"
                ):
                    if section.attrs["label"].decode() != "Elements_t":
                        continue
                    element_type_tag = int(zone[f"{section_name}/ data"][0])
                    if element_type_tag in [5, 7]:
                        names.append(f"{zone_name}/{section_name}")
                    if element_type_tag == 20:
                        first_element_type_tag = zone[f"{section_name}/ElementConnectivity/ data"][
                            0
                        ]
                        if first_element_type_tag in [5, 7]:
                            names.append(f"{zone_name}/{section_name}")
                else:
                    if section.attrs["label"].decode() != "ZoneBC_t":
                        continue
                    for bc_name, bc_zone in section.items():
                        if bc_zone.attrs["label"].decode() == "BC_t":
                            names.append(f"{zone_name}/{bc_name}")

        return names


def validate_cgns(
    cgns_file: str, params: Union[Flow360Params, Flow360MeshParams], solver_version=None
):
    """
    Validate CGNS file
    :param cgns_file:
    :param params:
    :param solver_version:
    :return:
    """
    assert cgns_file
    assert params
    boundaries_in_file = get_boundaries_from_file(cgns_file, solver_version)
    boundaries_in_params = get_no_slip_walls(params) + get_boundaries_from_sliding_interfaces(
        params
    )
    boundaries_in_file = set(boundaries_in_file)
    boundaries_in_params = set(boundaries_in_params)

    if not boundaries_in_file.issuperset(boundaries_in_params):
        raise Flow360ValueError(
            "The following input boundary names from mesh json are not found in mesh:"
            + f" {' '.join(boundaries_in_params - boundaries_in_file)}."
            + f" Boundary names in cgns: {' '.join(boundaries_in_file)}"
            + f" Boundary names in params: {' '.join(boundaries_in_file)}"
        )
    log.info(
        f'Notice: {" ".join(boundaries_in_file - boundaries_in_params)} is '
        + "tagged as wall in cgns file, but not in input params"
    )


class VolumeMeshLog(Enum):
    """
    Volume mesh log
    """

    USER_LOG = "user.log"
    PY_LOG = "validateFlow360Mesh.py.log"


class VolumeMeshDownloadable(Enum):
    """
    Volume mesh downloadable files
    """

    CONFIG_JSON = "config.json"
    BOUNDING_BOX = "meshBoundaryBoundingBox.json"


# pylint: disable=E0213
class VolumeMeshMeta(AssetMetaBaseModel, extra=pd.Extra.allow):
    """
    VolumeMeshMeta component
    """

    id: str = pd.Field(alias="meshId")
    name: str = pd.Field(alias="meshName")
    created_at: str = pd.Field(alias="meshAddTime")
    surface_mesh_id: Optional[str] = pd.Field(alias="surfaceMeshId")
    mesh_params: Union[Flow360MeshParams, None, dict] = pd.Field(alias="meshParams")
    mesh_format: Union[MeshFileFormat, None] = pd.Field(alias="meshFormat")
    file_name: Union[str, None] = pd.Field(alias="fileName")
    endianness: UGRIDEndianness = pd.Field(alias="meshEndianness")
    compression: CompressionFormat = pd.Field(alias="meshCompression")
    boundaries: Union[List, None]

    @pd.validator("mesh_params", pre=True)
    def init_mesh_params(cls, value):
        """
        validator for mesh_params
        """
        return params_generic_validator(value, Flow360MeshParams)

    @pd.validator("endianness", pre=True)
    def init_endianness(cls, value):
        """
        validator for endianess
        """
        return UGRIDEndianness(value) or UGRIDEndianness.NONE

    @pd.validator("compression", pre=True)
    def init_compression(cls, value):
        """
        validator for compression
        """
        try:
            return CompressionFormat(value)
        except ValueError:
            return CompressionFormat.NONE

    def to_volume_mesh(self) -> VolumeMesh:
        """
        returns VolumeMesh object from volume mesh meta info
        """
        return VolumeMesh(self.id)


class VolumeMeshDraft(ResourceDraft):
    """
    Volume mesh draft component (before submit)
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        file_name: str = None,
        params: Union[Flow360MeshParams, VolumeMeshingParams] = None,
        name: str = None,
        surface_mesh_id=None,
        tags: List[str] = None,
        solver_version=None,
        endianess: UGRIDEndianness = None,
        isascii: bool = False,
    ):
        if file_name is not None and not os.path.exists(file_name):
            raise Flow360FileError(f"File '{file_name}' not found.")

        if endianess is not None:
            raise Flow360NotImplementedError(
                "endianess selections not supported, it is inferred from filename"
            )

        if isascii is True:
            raise Flow360NotImplementedError("isascii not supported")

        self.params = None
        if params is not None:
            if not isinstance(params, Flow360MeshParams) and not isinstance(
                params, VolumeMeshingParams
            ):
                raise ValueError(
                    f"params={params} are not of type Flow360MeshParams OR VolumeMeshingParams"
                )
            self.params = params.copy(deep=True)

        if file_name is not None:
            mesh_parser = MeshNameParser(file_name)
            if not mesh_parser.is_valid_volume_mesh():
                raise Flow360ValueError(
                    f"Unsupported volume mesh file extensions: {mesh_parser.format.ext()}. "
                    f"Supported: [{MeshFileFormat.UGRID.ext()},{MeshFileFormat.CGNS.ext()}]."
                )

        if name is None and file_name is not None:
            name = os.path.splitext(os.path.basename(file_name))[0]

        self.file_name = file_name
        self.name = name
        self.surface_mesh_id = surface_mesh_id
        self.tags = tags
        self.solver_version = solver_version
        self._id = None
        self.compress_method = CompressionFormat.ZST
        ResourceDraft.__init__(self)

    def _submit_from_surface(self, force_submit: bool = False):
        self.validator_api(
            self.params, solver_version=self.solver_version, raise_on_error=(not force_submit)
        )
        body = {
            "name": self.name,
            "tags": self.tags,
            "surfaceMeshId": self.surface_mesh_id,
            "config": self.params.flow360_json(),
            "format": "cgns",
        }
        if Flags.beta_features() and self.params.version is not None:
            body["version"] = self.params.version

        if Flags.beta_features() and self.params.version is not None:
            if self.params.version == "v2":
                body["format"] = "aflr3"

        if self.solver_version:
            body["solverVersion"] = self.solver_version

        resp = RestApi(VolumeMeshInterface.endpoint).post(body)
        if not resp:
            return None

        info = VolumeMeshMeta(**resp)
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        mesh = VolumeMesh(self.id)
        log.info(f"VolumeMesh successfully submitted: {mesh.short_description()}")
        return mesh

    # pylint: disable=protected-access, too-many-locals
    def _submit_upload_mesh(self, progress_callback=None):
        assert os.path.exists(self.file_name)

        mesh_parser = MeshNameParser(self.file_name)
        mesh_format = mesh_parser.format
        endianness = mesh_parser.endianness
        original_compression = mesh_parser.compression

        compression = (
            original_compression
            if original_compression != CompressionFormat.NONE
            else self.compress_method
        )

        if mesh_format is MeshFileFormat.CGNS:
            remote_file_name = "volumeMesh"
        else:
            remote_file_name = "mesh"
        remote_file_name = (
            f"{remote_file_name}{endianness.ext()}{mesh_format.ext()}{compression.ext()}"
        )

        name = self.name
        if name is None:
            name = os.path.splitext(os.path.basename(self.file_name))[0]

        if Flags.beta_features() and self.params is not None:
            req = NewVolumeMeshRequest(
                name=name,
                file_name=remote_file_name,
                tags=self.tags,
                format=mesh_format.value,
                endianness=endianness.value,
                compression=compression.value,
                params=self.params,
                solver_version=self.solver_version,
                version=self.params.version,
            )
        else:
            req = NewVolumeMeshRequest(
                name=name,
                file_name=remote_file_name,
                tags=self.tags,
                format=mesh_format.value,
                endianness=endianness.value,
                compression=compression.value,
                params=self.params,
                solver_version=self.solver_version,
            )
        resp = RestApi(VolumeMeshInterface.endpoint).post(req.dict())
        if not resp:
            return None

        info = VolumeMeshMeta(**resp)
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        mesh = VolumeMesh(self.id)

        # parallel compress and upload
        if (
            original_compression == CompressionFormat.NONE
            and self.compress_method == CompressionFormat.BZ2
        ):
            upload_id = mesh.create_multipart_upload(remote_file_name)
            compress_and_upload_chunks(self.file_name, upload_id, mesh, remote_file_name)

        elif (
            original_compression == CompressionFormat.NONE
            and self.compress_method == CompressionFormat.ZST
        ):
            compressed_file_name = zstd_compress(self.file_name)
            mesh._upload_file(
                remote_file_name, compressed_file_name, progress_callback=progress_callback
            )
            os.remove(compressed_file_name)
        else:
            mesh._upload_file(remote_file_name, self.file_name, progress_callback=progress_callback)
        mesh._complete_upload(remote_file_name)

        log.info(f"VolumeMesh successfully uploaded: {mesh.short_description()}")
        return mesh

    def submit(self, progress_callback=None, force_submit: bool = False) -> VolumeMesh:
        """submit mesh to cloud

        Parameters
        ----------
        progress_callback : callback, optional
            Use for custom progress bar, by default None

        Returns
        -------
        VolumeMesh
            VolumeMesh object with id
        """

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        if self.file_name is not None:
            return self._submit_upload_mesh(progress_callback)

        if self.surface_mesh_id is not None and self.name is not None and self.params is not None:
            return self._submit_from_surface(force_submit=force_submit)

        raise Flow360ValueError(
            "You must provide volume mesh file for upload or surface mesh Id with meshing parameters."
        )

    @classmethod
    def validator_api(
        cls, params: VolumeMeshingParams, solver_version=None, raise_on_error: bool = True
    ):
        """
        validation api: validates surface meshing parameters before submitting
        """
        return Validator.VOLUME_MESH.validate(
            params, solver_version=solver_version, raise_on_error=raise_on_error
        )


class VolumeMesh(Flow360Resource):
    """
    Volume mesh component
    """

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=VolumeMeshInterface,
            meta_class=VolumeMeshMeta,
            id=id,
        )
        self.__mesh_params = None

    @classmethod
    def _from_meta(cls, meta: VolumeMeshMeta):
        validate_type(meta, "meta", VolumeMeshMeta)
        volume_mesh = cls(id=meta.id)
        volume_mesh._set_meta(meta)
        return volume_mesh

    @property
    def info(self) -> VolumeMeshMeta:
        return super().info

    @property
    def _mesh_params(self) -> Flow360MeshParams:
        """
        returns mesh params
        """
        if self.__mesh_params is None:
            self.__mesh_params = self.info.mesh_params
        return self.__mesh_params

    @property
    def no_slip_walls(self):
        """
        returns mesh no_slip_walls
        """
        if self._mesh_params is None:
            return None
        return self._mesh_params.boundaries.no_slip_walls

    @property
    def all_boundaries(self):
        """
        returns mesh no_slip_walls
        """
        return self.info.boundaries

    # pylint: disable=too-many-arguments,R0801
    def download_file(
        self,
        file_name: Union[str, VolumeMeshDownloadable],
        to_file=None,
        to_folder=".",
        overwrite: bool = True,
        progress_callback=None,
        **kwargs,
    ):
        """
        Download file from surface mesh
        :param file_name:
        :param to_file:
        :return:
        """
        if isinstance(file_name, VolumeMeshDownloadable):
            file_name = file_name.value
        return super()._download_file(
            file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
            progress_callback=progress_callback,
            **kwargs,
        )

    # pylint: disable=R0801
    def download(self, to_file=None, to_folder=".", overwrite: bool = True):
        """
        Download volume mesh file
        :param to_file:
        :return:
        """
        status = self.status
        if not status.is_final():
            log.warning(f"Cannot download file because status={status}")
            return None

        remote_file_name = self.info.file_name
        if remote_file_name is None:
            remote_file_name = self._remote_file_name()

        return super()._download_file(
            remote_file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
        )

    # pylint: disable=arguments-differ
    def _complete_upload(self, remote_file_name):
        """
        Complete volume mesh upload
        :return:
        """
        resp = self.post({}, method=f"completeUpload?fileName={remote_file_name}")
        self._info = VolumeMeshMeta(**resp)

    @classmethod
    def _interface(cls):
        return VolumeMeshInterface

    @classmethod
    def _meta_class(cls):
        """
        returns volume mesh meta info class: VolumeMeshMeta
        """
        return VolumeMeshMeta

    @classmethod
    def _params_ancestor_id_name(cls):
        """
        returns surfaceMeshId name
        """
        return "surfaceMeshId"

    @classmethod
    def from_cloud(cls, mesh_id: str):
        """
        Get volume mesh info from cloud
        :param mesh_id:
        :return:
        """
        return cls(mesh_id)

    def _remote_file_name(self):
        """
        mesh filename on cloud
        """

        remote_file_name = None
        for file in self.get_download_file_list():
            try:
                MeshNameParser(file["fileName"])
                remote_file_name = file["fileName"]
            except Flow360RuntimeError:
                continue

        if remote_file_name is None:
            raise Flow360CloudFileError(f"No volume mesh file found for id={self.id}")

        return remote_file_name

    @classmethod
    def from_file(
        cls,
        file_name: str,
        params: Union[Flow360MeshParams, None] = None,
        name: str = None,
        tags: List[str] = None,
        solver_version=None,
        endianess: UGRIDEndianness = None,
        isascii: bool = False,
    ) -> VolumeMeshDraft:
        """
        Upload volume mesh from file
        :param volume_mesh_name:
        :param file_name:
        :param params:
        :param tags:
        :param solver_version:
        :return:
        """
        return VolumeMeshDraft(
            file_name=file_name,
            name=name,
            tags=tags,
            solver_version=solver_version,
            params=params,
            endianess=endianess,
            isascii=isascii,
        )

    @classmethod
    def copy_from_example(
        cls,
        example_id: str,
        name: str = None,
    ) -> VolumeMesh:
        """
        Create a new volume mesh by copying from an example mesh identified by `example_id`.

        Parameters
        ----------
        example_id : str
            The unique identifier of the example volume mesh to copy from.
        name : str, optional
            The name to assign to the new volume mesh. If not provided, the name
            of the example volume mesh will be used.

        Returns
        -------
        VolumeMesh
            A new instance of VolumeMesh copied from the example mesh if successful.

        Examples
        --------
        >>> new_mesh = VolumeMesh.copy_from_example('example_id_123', name='New Mesh')
        """

        if name is None:
            eg_vm = cls(example_id)
            name = eg_vm.name
        req = CopyExampleVolumeMeshRequest(example_id=example_id, name=name)
        resp = RestApi(f"{VolumeMeshInterface.endpoint}/examples/copy").post(req.dict())
        if not resp:
            raise RuntimeError("Something went wrong when accessing example mesh.")

        info = VolumeMeshMeta(**resp)
        return cls(info.id)

    @classmethod
    def create(
        cls,
        name: str,
        params: VolumeMeshingParams,
        surface_mesh_id: str,
        tags: List[str] = None,
        solver_version=None,
    ) -> VolumeMeshDraft:
        """
        Create volume mesh from surface mesh
        """

        return VolumeMeshDraft(
            name=name,
            surface_mesh_id=surface_mesh_id,
            solver_version=solver_version,
            params=params,
            tags=tags,
        )

    def create_case(
        self,
        name: str,
        params: Flow360Params,
        tags: List[str] = None,
        solver_version: str = None,
    ) -> CaseDraft:
        """
        Create new case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return Case.create(
            name, params, volume_mesh_id=self.id, tags=tags, solver_version=solver_version
        )


class VolumeMeshList(Flow360ResourceListBase):
    """
    VolumeMesh List component
    """

    def __init__(
        self,
        surface_mesh_id: str = None,
        from_cloud: bool = True,
        include_deleted: bool = False,
        limit=100,
    ):
        super().__init__(
            ancestor_id=surface_mesh_id,
            from_cloud=from_cloud,
            include_deleted=include_deleted,
            limit=limit,
            resourceClass=VolumeMesh,
        )

    def filter(self):
        """
        flitering list, not implemented yet
        """
        raise NotImplementedError("Filters are not implemented yet")

        # resp = list(filter(lambda i: i['caseStatus'] != 'deleted', resp))

    # pylint: disable=useless-parent-delegation
    def __getitem__(self, index) -> VolumeMesh:
        """
        returns VolumeMeshMeta item of the list
        """
        return super().__getitem__(index)

    # pylint: disable=useless-parent-delegation
    def __iter__(self) -> Iterator[VolumeMesh]:
        return super().__iter__()


###==== V2 API version ===###


class VolumeMeshStatusV2(Enum):
    """Status of volume mesh resource, the is_final method is overloaded"""

    SUBMITTED = "submitted"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    COMPLETED = "completed"
    PENDING = "pending"
    GENERATING = "generating"

    def is_final(self):
        """
        Checks if status is final for volume mesh resource

        Returns
        -------
        bool
            True if status is final, False otherwise.
        """
        if self in [VolumeMeshStatusV2.COMPLETED]:
            return True
        return False


class VolumeMeshMetaV2(AssetMetaBaseModelV2):
    """
    VolumeMeshMetaV2 component
    """

    status: VolumeMeshStatusV2 = pd_v2.Field()  # Overshadowing to ensure correct is_final() method
    file_name: Optional[str] = pd_v2.Field(None, alias="fileName")


class VolumeMeshStats(pd_v2.BaseModel):
    """
    Mesh stats
    """

    n_nodes: int = pd_v2.Field(..., alias="nNodes")
    n_triangles: int = pd_v2.Field(..., alias="nTriangles")
    n_quadrilaterals: int = pd_v2.Field(..., alias="nQuadrilaterals")
    n_tetrahedron: int = pd_v2.Field(..., alias="nTetrahedron")
    n_prism: int = pd_v2.Field(..., alias="nPrism")
    n_pyramid: int = pd_v2.Field(..., alias="nPyramid")
    n_hexahedron: int = pd_v2.Field(..., alias="nHexahedron")
    n_tet_wedge: int = pd_v2.Field(..., alias="nTetWedge")


class VolumeMeshBoundingBox(PerEntityResultCSVModel):
    """
    VolumeMeshBoundingBox
    """

    remote_file_name: str = pd.Field(VolumeMeshDownloadable.BOUNDING_BOX.value, frozen=True)
    _variables: Optional[List[str]] = None

    @property
    def entities(self):
        """
        Returns list of entities (boundary names) available for this result
        """
        return self.values.keys()

    def _filtered_sum(self):
        pass

    def _get_range(self, df, min_key: str, max_key: str) -> float:
        if min_key not in df.index or max_key not in df.index:
            return 0.0
        min_val = df.loc[min_key].min()
        max_val = df.loc[max_key].max()
        return max_val - min_val

    @property
    def length(self) -> float:
        """
        Compute and return the length of the bounding box.

        The length is calculated as the difference between the maximum and minimum
        x-coordinate values from the bounding box data.

        Returns:
            float: The computed length along the x-axis.
        """
        df = self.as_dataframe()
        return self._get_range(df, "xmin", "xmax")

    @property
    def width(self) -> float:
        """
        Compute and return the width of the bounding box.

        The width is calculated as the difference between the maximum and minimum
        y-coordinate values from the bounding box data.

        Returns:
            float: The computed width along the y-axis.
        """
        df = self.as_dataframe()
        return self._get_range(df, "ymin", "ymax")

    @property
    def height(self) -> float:
        """
        Compute and return the height of the bounding box.

        The height is calculated as the difference between the maximum and minimum
        z-coordinate values from the bounding box data.

        Returns:
            float: The computed height along the z-axis.
        """
        df = self.as_dataframe()
        return self._get_range(df, "zmin", "zmax")


class VolumeMeshDraftV2(ResourceDraft):
    """
    Volume mesh draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        file_names: str,
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ):
        self.file_name = file_names
        self.project_name = project_name
        self.tags = tags if tags is not None else []
        self.length_unit = length_unit
        self.solver_version = solver_version
        self._validate()
        ResourceDraft.__init__(self)

    def _validate(self):
        self._validate_volume_mesh()

    def _validate_volume_mesh(self):
        if self.file_name is not None:
            try:
                VolumeMeshFile(file_names=self.file_name)
            except pd.ValidationError as e:
                raise Flow360FileError(str(e)) from e

        if self.project_name is None:
            self.project_name = os.path.splitext(os.path.basename(self.file_name))[0]
            log.warning(
                "`project_name` is not provided. "
                f"Using the volume mesh file name {self.project_name} as project name."
            )

        if self.length_unit not in LengthUnitType.__args__:
            raise Flow360ValueError(
                f"specified length_unit : {self.length_unit} is invalid. "
                f"Valid options are: {list(LengthUnitType.__args__)}"
            )

        if self.solver_version is None:
            raise Flow360ValueError("solver_version field is required.")

    # pylint: disable=protected-access, too-many-locals
    def submit(
        self, description="", progress_callback=None, compress=True, run_async=False
    ) -> VolumeMeshV2:
        """
        Submit volume mesh to cloud and create a new project

        Parameters
        ----------
        description : str, optional
            description of the project, by default ""
        progress_callback : callback, optional
            Use for custom progress bar, by default None
        compress : boolean, optional
            Compress the volume mesh file when sending to S3, default is True
        fetch_entities : boolean, optional
            Whether to fetch and populate the entity info object after submitting, default is False
        run_async : bool, optional
            Whether to submit volume mesh asynchronously (default is False).

        Returns
        -------
        VolumeMeshV2
            Volume mesh object with id
        """

        self._validate()

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        mesh_parser = MeshNameParser(self.file_name)

        original_compression = mesh_parser.compression
        mesh_format = mesh_parser.format
        file_name_no_compression = mesh_parser.file_name_no_compression

        compression = original_compression
        do_compression = False
        if compress and original_compression == CompressionFormat.NONE:
            compression = CompressionFormat.ZST
            do_compression = True

        original_file_with_compression = f"{file_name_no_compression}{compression.ext()}"

        req = NewVolumeMeshRequestV2(
            name=self.project_name,
            solver_version=self.solver_version,
            tags=self.tags,
            file_name=original_file_with_compression,
            length_unit=self.length_unit,
            format=mesh_format.value,
            description=description,
        )

        # Create new volume mesh resource and project
        req_dict = req.dict()
        resp = RestApi(VolumeMeshInterfaceV2.endpoint).post(req_dict)
        info = VolumeMeshMetaV2(**resp)
        # setting _id will disable "WARNING: You have not submitted..." warning message
        self._id = info.id
        renamed_file_on_remote = info.file_name

        volume_mesh = VolumeMeshV2(info.id)

        # Upload volume mesh file, keep posting the heartbeat to keep server patient about uploading.
        heartbeat_info = {"resourceId": info.id, "resourceType": "VolumeMesh", "stop": False}
        heartbeat_thread = threading.Thread(target=post_upload_heartbeat, args=(heartbeat_info,))
        heartbeat_thread.start()

        # Compress (if not already compressed) and upload
        if do_compression:
            zstd_compress(self.file_name, original_file_with_compression)
            volume_mesh._webapi._upload_file(
                renamed_file_on_remote,
                original_file_with_compression,
                progress_callback=progress_callback,
            )
            os.remove(original_file_with_compression)
        else:
            volume_mesh._webapi._upload_file(
                renamed_file_on_remote, self.file_name, progress_callback=progress_callback
            )

        if mesh_parser.is_ugrid():
            expected_local_mapbc_file = mesh_parser.get_associated_mapbc_filename()
            if os.path.isfile(expected_local_mapbc_file):
                remote_mesh_parser = MeshNameParser(renamed_file_on_remote)
                volume_mesh._webapi._upload_file(
                    remote_mesh_parser.get_associated_mapbc_filename(),
                    mesh_parser.get_associated_mapbc_filename(),
                    progress_callback=progress_callback,
                )
            else:
                log.warning(
                    f"The expected mapbc file {expected_local_mapbc_file} specifying "
                    "user-specified boundary names doesn't exist."
                )

        heartbeat_info["stop"] = True
        heartbeat_thread.join()

        # Start processing pipeline
        volume_mesh._webapi._complete_upload()
        self._id = info.id
        if run_async:
            return volume_mesh
        log.debug("Waiting for volume mesh to be processed.")
        volume_mesh._webapi.get_info()
        log.info(f"VolumeMesh successfully submitted: {volume_mesh._webapi.short_description()}")

        return VolumeMeshV2.from_cloud(volume_mesh.id)


class VolumeMeshV2(AssetBase):
    """
    Volume mesh component for workbench (simulation V2)
    """

    _interface_class = VolumeMeshInterfaceV2
    _meta_class = VolumeMeshMetaV2
    _draft_class = VolumeMeshDraftV2
    _web_api_class = Flow360Resource
    _entity_info_class = VolumeMeshEntityInfo
    _mesh_stats_file = "meshStats.json"
    _cloud_resource_type_name = "VolumeMesh"

    @classmethod
    # pylint: disable=redefined-builtin
    def from_cloud(cls, id: str, **kwargs) -> VolumeMeshV2:
        """
        Parameters
        ----------
        id : str
            ID of the volume mesh resource in the cloud

        Returns
        -------
        VolumeMeshV2
            Volume mesh object
        """
        asset_obj = super().from_cloud(id, **kwargs)

        return asset_obj

    @classmethod
    def from_local_storage(
        cls, mesh_id: str = None, local_storage_path="", meta_data: VolumeMeshMetaV2 = None
    ) -> VolumeMeshV2:
        """
        Parameters
        ----------
        mesh_id : str
            ID of the volume mesh resource

        local_storage_path:
            The folder of the project, defaults to current working directory

        Returns
        -------
        VolumeMeshV2
            Volume mesh object
        """
        return super()._from_local_storage(
            asset_id=mesh_id, local_storage_path=local_storage_path, meta_data=meta_data
        )

    @classmethod
    # pylint: disable=too-many-arguments,arguments-renamed
    def from_file(
        cls,
        file_name: str,
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ) -> VolumeMeshDraftV2:
        """
        Parameters
        ----------
        file_name : str
            The name of the input volume mesh file (*.cgns, *.ugrid)
        project_name : str, optional
            The name of the newly created project, defaults to file name if empty
        solver_version: str
            Solver version to use for the project
        length_unit: LengthUnitType
            Length unit to use for the project ("m", "mm", "cm", "inch", "ft")
        tags: List[str]
            List of string tags to be added to the project upon creation

        Returns
        -------
        VolumeMeshDraftV2
            Draft of the volume mesh to be submitted
        """
        # For type hint only but proper fix is to fully abstract the Draft class too.
        return super().from_file(
            file_names=file_name,
            project_name=project_name,
            solver_version=solver_version,
            length_unit=length_unit,
            tags=tags,
        )

    # pylint: disable=useless-parent-delegation
    def get_default_settings(self, simulation_dict: dict):
        """Get the default volume mesh settings from the simulation dict"""
        return super().get_default_settings(simulation_dict)

    @cached_property
    def stats(self) -> VolumeMeshStats:
        """
        Get mesh stats

        Returns
        -------
        VolumeMeshStats
            return VolumeMeshStats object
        """
        # pylint: disable=protected-access
        data = self._webapi._parse_json_from_cloud(self._mesh_stats_file)
        return VolumeMeshStats(**data)

    @cached_property
    def bounding_box(self) -> VolumeMeshBoundingBox:
        """
        Get mesh bounding box

        Returns
        -------
        VolumeMeshBoundingBox
            return VolumeMeshBoundingBox object
        """

        # pylint: disable=protected-access
        data = self._webapi._parse_json_from_cloud(VolumeMeshDownloadable.BOUNDING_BOX.value)
        bbox = VolumeMeshBoundingBox.from_dict(data)
        return bbox

    @property
    def boundary_names(self) -> List[str]:
        """
        Retrieve all boundary names available in this volume mesh as a list

        Returns
        -------
        List[str]
            List of boundary names contained within the volume mesh
        """
        self.internal_registry = self._entity_info.get_registry(
            internal_registry=self.internal_registry
        )

        return [
            surface.name for surface in self.internal_registry.get_bucket(by_type=Surface).entities
        ]

    @property
    def zone_names(self) -> List[str]:
        """
        Retrieve all volume zone names available in this volume mesh as a list

        Returns
        -------
        List[str]
            List of zone names contained within the volume mesh
        """
        self.internal_registry = self._entity_info.get_registry(
            internal_registry=self.internal_registry
        )

        return [
            volume.name
            for volume in self.internal_registry.get_bucket(by_type=GenericVolume).entities
        ]

    def __getitem__(self, key: str):
        """
        Parameters
        ----------
        key : str
            The name of the entity to be found

        Returns
        -------
        Surface
            The boundary object
        """
        if isinstance(key, str) is False:
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        self.internal_registry = self._entity_info.get_registry(
            internal_registry=self.internal_registry
        )

        return self.internal_registry.find_by_naming_pattern(
            key, enforce_output_as_list=False, error_when_no_match=True
        )

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError("Assigning/setting entities is not supported.")
