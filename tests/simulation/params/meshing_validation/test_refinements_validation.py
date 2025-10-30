import re

import pydantic as pd
import pytest

import flow360.component.simulation.units as u

from flow360.component.simulation.meshing_param.snappy import (
    SnappyRegionRefinement,
    SnappySurfaceEdgeRefinement,
    SnappySurfaceMeshingDefaults,
    SnappySurfaceMeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Box, Cylinder, Surface
from flow360.component.simulation.unit_system import SI_unit_system


def test_snappy_refinements_validators():
    message = "Minimum spacing must be lower than maximum spacing."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SnappyRegionRefinement(
            min_spacing=4.3 * u.mm, max_spacing=2.1 * u.mm, regions=[Surface(name="test")]
        )

    message = "UniformRefinement for snappy accepts only Boxes with axes aligned with the global coordinate system (angle_of_rotation=0)."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)) as err:
        SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=10 * u.mm, gap_resolution=0.1 * u.mm
            ),
            refinements=[
                UniformRefinement(
                    name="unif",
                    spacing=6 * u.mm,
                    entities=[
                        Box(
                            center=[2, 3, 4] * u.m,
                            size=[5, 6, 7] * u.m,
                            axis_of_rotation=[1, 3, 4],
                            angle_of_rotation=5 * u.deg,
                            name="box",
                        )
                    ],
                )
            ],
        )

    SnappySurfaceMeshingParams(
        defaults=SnappySurfaceMeshingDefaults(
            min_spacing=3 * u.mm, max_spacing=10 * u.mm, gap_resolution=0.1 * u.mm
        ),
        refinements=[
            UniformRefinement(
                name="unif",
                spacing=6 * u.mm,
                entities=[
                    Box(
                        center=[2, 3, 4] * u.m,
                        size=[5, 6, 7] * u.m,
                        axis_of_rotation=[1, 3, 4],
                        angle_of_rotation=0 * u.deg,
                        name="box",
                    )
                ],
            )
        ],
    )

    message = "UniformRefinement for snappy accepts only full cylinders (where inner_radius = 0)."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=10 * u.mm, gap_resolution=0.1 * u.mm
            ),
            refinements=[
                UniformRefinement(
                    name="unif",
                    spacing=6 * u.mm,
                    entities=[
                        Cylinder(
                            name="cyl",
                            inner_radius=3 * u.mm,
                            outer_radius=7 * u.mm,
                            axis=[0, 0, 1],
                            center=[0, 0, 0] * u.m,
                            height=10 * u.mm,
                        )
                    ],
                )
            ],
        )


def test_snappy_edge_refinement_valdators():
    with pytest.raises(ValueError):
        SnappySurfaceEdgeRefinement(
            spacing=2 * u.mm, distances=[5 * u.mm], regions=[Surface(name="test")]
        )

    with pytest.raises(ValueError):
        SnappySurfaceEdgeRefinement(
            spacing=[2 * u.mm, 3 * u.mm], distances=[5 * u.mm], regions=[Surface(name="test")]
        )

    with pytest.raises(ValueError):
        SnappySurfaceEdgeRefinement(
            spacing=2 * u.mm, distances=5 * u.mm, regions=[Surface(name="test")]
        )

    with pytest.raises(ValueError):
        SnappySurfaceEdgeRefinement(spacing=[2 * u.mm], regions=[Surface(name="test")])

    SnappySurfaceEdgeRefinement(
        spacing=[2 * u.mm], distances=[5 * u.mm], regions=[Surface(name="test")]
    )

    SnappySurfaceEdgeRefinement(regions=[Surface(name="test")])

    SnappySurfaceEdgeRefinement(spacing=2 * u.mm, regions=[Surface(name="test")])

    SnappySurfaceMeshingParams(
        defaults=SnappySurfaceMeshingDefaults(
            min_spacing=3 * u.mm, max_spacing=6 * u.mm, gap_resolution=0.1 * u.mm
        ),
        refinements=[
            SnappySurfaceEdgeRefinement(
                spacing=[2 * u.mm], distances=[5 * u.mm], regions=[Surface(name="test")]
            ),
            SnappySurfaceEdgeRefinement(spacing=2 * u.mm, regions=[Surface(name="test")]),
            SnappySurfaceEdgeRefinement(regions=[Surface(name="test")]),
        ],
    )
