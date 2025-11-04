"""surface meshing parameters to use with snappyHexMesh"""

from typing import List, Literal, Optional

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.meshing_specs import OctreeSpacing
from flow360.component.simulation.meshing_param.snappy.snappy_mesh_refinements import (
    BodyRefinement,
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
from flow360.component.simulation.primitives import Box, Cylinder
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
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
    smooth_controls: Optional[SmoothControls] = pd.Field(None)
    refinements: Optional[List[SnappySurfaceRefinementTypes]] = pd.Field(None)
    base_spacing: Optional[OctreeSpacing] = pd.Field(None)

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
                        "Default minimum spacing is higher that refinement maximum spacing"
                        + "and minimum spacing is not provided."
                    )
                if refinement.max_spacing is None and self.defaults.max_spacing.to(
                    "m"
                ) < refinement.min_spacing.to("m"):
                    raise ValueError(
                        "Default maximum spacing is lower that refinement minimum spacing"
                        + "and maximum spacing is not provided."
                    )
        return self

    @pd.model_validator(mode="after")
    def _check_uniform_refinement_entities(self):
        # pylint: disable=no-member
        if self.refinements is None:
            return self
        for refinement in self.refinements:
            if isinstance(refinement, UniformRefinement):
                for entity in refinement.entities.stored_entities:
                    if isinstance(entity, Box) and entity.angle_of_rotation.to("deg") != 0 * u.deg:
                        raise ValueError(
                            "UniformRefinement for snappy accepts only Boxes with axes aligned"
                            + " with the global coordinate system (angle_of_rotation=0)."
                        )
                    if isinstance(entity, Cylinder) and entity.inner_radius.to("m") != 0 * u.m:
                        raise ValueError(
                            "UniformRefinement for snappy accepts only full cylinders (where inner_radius = 0)."
                        )

        return self

    @pd.model_validator(mode="after")
    def _check_sizing_against_octree_series(self):

        if self.base_spacing is None:
            return self

        def check_spacing(spacing, location):
            # pylint: disable=no-member
            lvl, close = self.base_spacing.to_level(spacing)
            spacing_unit = spacing.units
            if not close:
                closest_spacing = self.base_spacing[lvl]
                msg = f"The spacing of {spacing:.4g} spcified in {location} will be cast to the first lower refinement"
                msg += f" in the octree series which is {closest_spacing.to(spacing_unit):.4g}."
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

    @pd.field_validator("base_spacing", mode="after")
    @classmethod
    def _set_default_base_spacing(cls, base_spacing):
        info = get_validation_info()
        if (info is None) or (base_spacing is not None) or (info.project_length_unit is None):
            return base_spacing

        # pylint: disable=no-member
        base_spacing = 1 * LengthType.validate(info.project_length_unit)
        return OctreeSpacing(base_spacing=base_spacing)
