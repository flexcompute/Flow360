"""Asset cache payload models for simulation-side private cache serialization."""

from typing import Any, Literal

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_operation import CoordinateSystem
from flow360_schema.framework.entity.entity_selector import EntitySelector
from flow360_schema.framework.physical_dimensions import Length
from flow360_schema.models.entities.geometry_entities import MirrorPlane
from flow360_schema.models.entities.surface_entities import (
    ImportedSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
)
from flow360_schema.models.entity_info import (
    GeometryEntityInfo,
    SurfaceMeshEntityInfo,
    VolumeMeshEntityInfo,
)
from flow360_schema.models.variable_context import VariableContextList


class MirrorStatus(Flow360BaseModel):
    """
    Serializable snapshot of mirror state stored in the asset cache.

    This status stores both user-authored mirror inputs and derived mirrored entities.
    """

    mirror_planes: list[MirrorPlane] = pd.Field(description="List of mirror planes to mirror.")
    mirrored_geometry_body_groups: list[MirroredGeometryBodyGroup] = pd.Field(
        description="List of mirrored geometry body groups."
    )
    mirrored_surfaces: list[MirroredSurface] = pd.Field(description="List of mirrored surfaces.")

    @pd.model_validator(mode="after")
    def _validate_unique_mirror_plane_names(self) -> "MirrorStatus":
        """Validate that all mirror plane names are unique."""
        seen_names = set()
        for plane in self.mirror_planes:
            if plane.name in seen_names:
                raise ValueError(f"Duplicate mirror plane name '{plane.name}' found in mirror status.")
            seen_names.add(plane.name)
        return self

    def is_empty(self) -> bool:
        """Return True when no mirroring is configured."""
        return not self.mirror_planes and not self.mirrored_geometry_body_groups and not self.mirrored_surfaces


class CoordinateSystemParent(Flow360BaseModel):
    """Parent relationship for a coordinate system."""

    type_name: Literal["CoordinateSystemParent"] = pd.Field("CoordinateSystemParent", frozen=True)
    coordinate_system_id: str
    parent_id: str | None = pd.Field(None)


class CoordinateSystemEntityRef(Flow360BaseModel):
    """Entity reference used in coordinate-system assignment serialization."""

    type_name: Literal["CoordinateSystemEntityRef"] = pd.Field("CoordinateSystemEntityRef", frozen=True)
    entity_type: str
    entity_id: str


class CoordinateSystemAssignmentGroup(Flow360BaseModel):
    """Grouped entity assignments for a coordinate system."""

    type_name: Literal["CoordinateSystemAssignmentGroup"] = pd.Field("CoordinateSystemAssignmentGroup", frozen=True)
    coordinate_system_id: str
    entities: list[CoordinateSystemEntityRef]


class CoordinateSystemStatus(Flow360BaseModel):
    """Serializable snapshot of draft coordinate systems and assignments."""

    type_name: Literal["CoordinateSystemStatus"] = pd.Field("CoordinateSystemStatus", frozen=True)
    coordinate_systems: list[CoordinateSystem]
    parents: list[CoordinateSystemParent]
    assignments: list[CoordinateSystemAssignmentGroup]

    @pd.model_validator(mode="after")
    def _validate_unique_coordinate_system_ids_and_names(self) -> "CoordinateSystemStatus":
        """Validate that all coordinate system IDs and names are unique."""
        seen_ids = set()
        seen_names = set()
        for coordinate_system in self.coordinate_systems:
            if coordinate_system.private_attribute_id in seen_ids:
                raise ValueError(
                    f"[Internal] Duplicate coordinate system id '{coordinate_system.private_attribute_id}' in status."
                )
            if coordinate_system.name in seen_names:
                raise ValueError(f"[Internal] Duplicate coordinate system name '{coordinate_system.name}' in status.")
            seen_ids.add(coordinate_system.private_attribute_id)
            seen_names.add(coordinate_system.name)
        return self


class AssetCache(Flow360BaseModel):
    """
    Cached info from the project asset.
    """

    project_length_unit: Length.PositiveFloat64 | None = pd.Field(None, frozen=True)  # type: ignore[valid-type]
    project_entity_info: GeometryEntityInfo | VolumeMeshEntityInfo | SurfaceMeshEntityInfo | None = pd.Field(
        None, frozen=True, discriminator="type_name"
    )
    use_inhouse_mesher: bool = pd.Field(
        False,
        description="Flag whether user requested the use of inhouse surface and volume mesher.",
    )
    use_geometry_AI: bool = pd.Field(False, description="Flag whether user requested the use of GAI.")
    # FXC-3289: which CAD Importer version the producer ran for this project.
    # The CAD Importer is the producer-side pipeline that turns user CAD into the
    # face/edge partition the mesher consumes.
    #   - "v1" (default, HOOPS -> EGADS): HOOPS reads the CAD and convertSTEPToEGADS
    #     healing produces the EGADS face partition; CADToGeometry
    #     --legacy-compatibility then emits indexed legacy ids. Compatible with all
    #     meshers (legacy surface mesher, beta in-house mesher, Geometry AI).
    #   - "v2" (HOOPS only): convertSTEPToEGADS is skipped; CADToGeometry
    #     --legacy-compatibility owns the face partition directly via HOOPS' native
    #     BREP analysis and re-exports STEP with the indexed legacy ids stamped as
    #     Name attributes. Currently compatible only with the legacy surface mesher
    #     -- the beta in-house mesher and Geometry AI both need the v1 .egads file.
    #
    # Frozen at geometry upload; immutable for the project lifetime. Choice is
    # per-CAD-file -- there is no across-the-board "better" engine; try v2 when v1
    # doesn't handle a particular file, and vice versa.
    cad_importer_version: Literal["v1", "v2"] = pd.Field(
        "v1",
        frozen=True,
        description=(
            "CAD Importer version frozen at geometry upload. "
            "'v1' (default) is compatible with all surface meshers; "
            "'v2' is an alternative BRep importer that currently supports "
            "only the legacy surface mesher."
        ),
    )
    variable_context: VariableContextList | None = pd.Field(
        None,
        description="List of user variables that are used in all the `Expression` instances.",
    )
    used_selectors: list[EntitySelector] | None = pd.Field(
        None,
        description="Collected entity selectors for token reference.",
    )
    imported_surfaces: list[ImportedSurface] | None = pd.Field(
        None, description="List of imported surface meshes for post-processing."
    )
    mirror_status: MirrorStatus | None = pd.Field(
        None, description="Status of mirroring operations that are used in the simulation."
    )
    coordinate_system_status: CoordinateSystemStatus | None = pd.Field(
        None, description="Status of coordinate systems used in the simulation."
    )

    @property
    def boundaries(self) -> list[Surface] | None:
        """
        Get all boundaries (not just names) from the cached entity info.
        """
        if self.project_entity_info is None:
            return None
        return self.project_entity_info.get_boundaries()

    @pd.model_validator(mode="after")
    def _validate_cad_importer_mesher_compatibility(self) -> "AssetCache":
        """
        CAD Importer v2 (HOOPS only) produces a HOOPS-native face partition and
        never emits the .egads file v1 produces.

        The beta in-house *surface* mesher on its own (`use_inhouse_mesher`
        without Geometry AI) reads that EGADS face partition directly and would
        crash mid-run when no .egads is available, so it stays incompatible with
        v2.

        Geometry AI (`use_geometry_AI`) is supported on v2 -- including when the
        beta mesher flag is also set, which is the usual GAI configuration. The
        GAI surface mesher re-tessellates the stamped v2 STEP through the HOOPS
        importer (Surf360 --useHOOPS), which supplies the face partition in place
        of the EGADS one, with the legacy STEP-"Name" ids matching
        project_entity_info.
        """
        if self.cad_importer_version != "v2":
            return self
        if self.use_inhouse_mesher and not self.use_geometry_AI:
            raise ValueError(
                "Beta mesher requires CAD Importer V1. Re-upload this project with CAD Importer V1 to enable."
            )
        return self

    @pd.model_validator(mode="after")
    def _validate_mirror_status_compatible_with_geometry(self) -> "AssetCache":
        """Raise if mirror_status has mirroring but geometry doesn't support face-to-body-group mapping."""
        if self.mirror_status is None:
            return self
        if not self.mirror_status.mirrored_geometry_body_groups:
            return self
        if not isinstance(self.project_entity_info, GeometryEntityInfo):
            return self

        try:
            self.project_entity_info.get_face_group_to_body_group_id_map()
        except ValueError as exc:
            raise ValueError(
                "Mirroring is requested but the geometry's face groupings span across body groups. "
                f"Mirroring cannot be performed: {exc}"
            ) from exc
        return self

    def preprocess(
        self,
        *,
        params: Any = None,
        exclude: list[str] | None = None,
        required_by: list[str] | None = None,
        flow360_unit_system: Any = None,
    ) -> Flow360BaseModel:
        # Exclude variable_context and used_selectors from preprocessing.
        # NOTE: coordinate_system_status is NOT excluded, which means it will be
        # recursively preprocessed. This is CRITICAL because CoordinateSystem objects
        # contain LengthType fields (origin, translation) that must be nondimensionalized
        # before transformation matrices are computed in the translator.
        exclude_asset_cache = (exclude or []) + ["variable_context", "used_selectors"]
        return super().preprocess(
            params=params,
            exclude=exclude_asset_cache,
            required_by=required_by,
            flow360_unit_system=flow360_unit_system,
        )


__all__ = [
    "AssetCache",
    "CoordinateSystemAssignmentGroup",
    "CoordinateSystemEntityRef",
    "CoordinateSystemParent",
    "CoordinateSystemStatus",
    "MirrorStatus",
]
