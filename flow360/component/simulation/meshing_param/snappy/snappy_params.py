"""surface meshing parameters to use with snappyHexMesh"""

from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.meshing_specs import OctreeSpacing
from flow360.component.simulation.meshing_param.meshing_validators import (
    validate_snappy_uniform_refinement_entities,
)
from flow360.component.simulation.meshing_param.snappy.snappy_mesh_refinements import (
    BodyRefinement,
    RegionRefinement,
    SnappyEntityRefinement,
    SnappySurfaceRefinementTypes,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.snappy.snappy_specs import (
    CastellatedMeshControls,
    QualityMetrics,
    SmoothControls,
    SnapControls,
    SurfaceMeshingDefaults,
)
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_field_validator,
    contextual_model_validator,
)
from flow360.log import log


class SurfaceMeshingParams(Flow360BaseModel):
    """
    Parameters for snappyHexMesh surface meshing.
    """

    type_name: Literal["SnappySurfaceMeshingParams"] = pd.Field(
        "SnappySurfaceMeshingParams", frozen=True
    )
    defaults: SurfaceMeshingDefaults = pd.Field()
    quality_metrics: QualityMetrics = pd.Field(QualityMetrics())
    snap_controls: SnapControls = pd.Field(SnapControls())
    castellated_mesh_controls: CastellatedMeshControls = pd.Field(CastellatedMeshControls())
    smooth_controls: Union[SmoothControls, Literal[False]] = pd.Field(SmoothControls())
    refinements: Optional[List[SnappySurfaceRefinementTypes]] = pd.Field(None)
    octree_spacing: Optional[OctreeSpacing] = pd.Field(None, validation_alias="base_spacing")

    @pd.model_validator(mode="before")
    @classmethod
    def _warn_base_spacing_deprecated(cls, data):
        if isinstance(data, dict) and "base_spacing" in data:
            log.warning(
                "`base_spacing` has been renamed to `octree_spacing`. "
                "Please update your code. `base_spacing` will be removed in a future release."
            )
        return data

    @pd.model_validator(mode="after")
    def _check_body_refinements_w_defaults(self):
        # set body refinements
        # pylint: disable=no-member
        if self.refinements is None:
            return self
        for refinement in self.refinements:
            if isinstance(refinement, BodyRefinement):
                if refinement.min_spacing is None and refinement.max_spacing is None:
                    continue
                if refinement.min_spacing is None and self.defaults.min_spacing.to(
                    "m"
                ) > refinement.max_spacing.to("m"):
                    raise ValueError(
                        f"Default minimum spacing ({self.defaults.min_spacing}) is higher than "
                        + f"refinement maximum spacing ({refinement.max_spacing}) "
                        + "and minimum spacing is not provided for BodyRefinement."
                    )
                if refinement.max_spacing is None and self.defaults.max_spacing.to(
                    "m"
                ) < refinement.min_spacing.to("m"):
                    raise ValueError(
                        f"Default maximum spacing ({self.defaults.max_spacing}) is lower than "
                        + f"refinement minimum spacing ({refinement.min_spacing}) "
                        + "and maximum spacing is not provided for BodyRefinement."
                    )
        return self

    @pd.field_validator("refinements", mode="after")
    @classmethod
    def _check_duplicate_refinements_per_entity(cls, refinements):
        """Raise if the same refinement type is applied more than once to the same entity."""
        if refinements is None:
            return refinements

        entity_refinement_map: dict[tuple[str, str], dict[str, int]] = {}
        refinement_types_with_entities = (BodyRefinement, RegionRefinement, SurfaceEdgeRefinement)

        for refinement in refinements:
            if not isinstance(refinement, refinement_types_with_entities):
                continue
            if refinement.entities is None:
                continue
            refinement_type_name = type(refinement).__name__
            for entity in refinement.entities.stored_entities:
                entity_key = (type(entity).__name__, entity.name)
                counts = entity_refinement_map.setdefault(entity_key, {})
                counts[refinement_type_name] = counts.get(refinement_type_name, 0) + 1

        for entity_key, type_counts in entity_refinement_map.items():
            for refinement_type_name, count in type_counts.items():
                if count > 1:
                    raise ValueError(
                        f"`{refinement_type_name}` is applied {count} times "
                        f"to entity `{entity_key[1]}`. Each refinement type "
                        f"can only be applied once per entity."
                    )
        return refinements

    @contextual_model_validator(mode="after")
    def _check_uniform_refinement_entities(self):
        # pylint: disable=no-member
        if self.refinements is None:
            return self
        for refinement in self.refinements:
            if isinstance(refinement, UniformRefinement):
                validate_snappy_uniform_refinement_entities(refinement)

        return self

    @pd.model_validator(mode="after")
    def _check_sizing_against_octree_series(self):

        if self.octree_spacing is None:
            return self

        def check_spacing(spacing, location):
            # pylint: disable=no-member
            lvl, close = self.octree_spacing.to_level(spacing)
            spacing_unit = spacing.units
            if not close:
                closest_spacing = self.octree_spacing[lvl]
                msg = f"The spacing of {spacing:.4g} specified in {location} will be cast to the first lower refinement"
                msg += f" in the octree series ({closest_spacing.to(spacing_unit):.4g})."
                log.warning(msg)

        # pylint: disable=no-member
        check_spacing(self.defaults.min_spacing, "defaults")
        check_spacing(self.defaults.max_spacing, "defaults")

        if self.refinements is not None:
            # pylint: disable=not-an-iterable
            for refinement in self.refinements:
                if isinstance(refinement, SnappyEntityRefinement):
                    if refinement.min_spacing is not None:
                        check_spacing(refinement.min_spacing, type(refinement).__name__)
                    if refinement.max_spacing is not None:
                        check_spacing(refinement.max_spacing, type(refinement).__name__)
                    if refinement.proximity_spacing is not None:
                        check_spacing(refinement.proximity_spacing, type(refinement).__name__)
                if isinstance(refinement, SurfaceEdgeRefinement):
                    if refinement.distances is not None:
                        for spacing in refinement.spacing:
                            check_spacing(spacing, type(refinement).__name__)
                    else:
                        check_spacing(refinement.spacing, type(refinement).__name__)
                if isinstance(refinement, UniformRefinement):
                    check_spacing(refinement.spacing, type(refinement).__name__)

        return self

    @contextual_field_validator("octree_spacing", mode="after")
    @classmethod
    def _set_default_octree_spacing(cls, octree_spacing, param_info: ParamsValidationInfo):
        if (octree_spacing is not None) or (param_info.project_length_unit is None):
            return octree_spacing

        # pylint: disable=no-member
        project_length = 1 * LengthType.validate(param_info.project_length_unit)
        return OctreeSpacing(base_spacing=project_length)
