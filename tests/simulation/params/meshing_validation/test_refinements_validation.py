import re

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Box, Cylinder, SnappyBody, Surface
from flow360.component.simulation.unit_system import SI_unit_system


def test_snappy_refinements_validators(mock_validation_context):
    message = "Minimum spacing must be lower than maximum spacing."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.RegionRefinement(
            min_spacing=4.3 * u.mm, max_spacing=2.1 * u.mm, regions=[Surface(name="test")]
        )

    message = "UniformRefinement for snappy accepts only Boxes with axes aligned with the global coordinate system (angle_of_rotation=0)."
    with (
        mock_validation_context,
        SI_unit_system,
        pytest.raises(ValueError, match=re.escape(message)),
    ):
        snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
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

    snappy.SurfaceMeshingParams(
        defaults=snappy.SurfaceMeshingDefaults(
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

    snappy.SurfaceMeshingParams(
        defaults=snappy.SurfaceMeshingDefaults(
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
                        angle_of_rotation=360 * u.deg,
                        name="box",
                    )
                ],
            )
        ],
    )

    message = "UniformRefinement for snappy accepts only full cylinders (where inner_radius = 0)."
    with (
        mock_validation_context,
        SI_unit_system,
        pytest.raises(ValueError, match=re.escape(message)),
    ):
        snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
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


def test_snappy_edge_refinement_validators():
    message = "When using a distance spacing specification both spacing (2.0 mm) and distances ([5] mm) fields must be arrays and the same length."
    with pytest.raises(
        ValueError,
        match=re.escape(message),
    ):
        snappy.SurfaceEdgeRefinement(
            spacing=2 * u.mm, distances=[5 * u.mm], entities=[Surface(name="test")]
        )

    with pytest.raises(
        pd.ValidationError,
    ):
        snappy.SurfaceEdgeRefinement(
            spacing=[2 * u.mm, 3 * u.mm], distances=[5 * u.mm], entities=[Surface(name="test")]
        )

    with pytest.raises(pd.ValidationError):
        snappy.SurfaceEdgeRefinement(
            spacing=2 * u.mm, distances=5 * u.mm, entities=[Surface(name="test")]
        )

    message = "When using a distance spacing specification both spacing ([2] mm) and distances (None) fields must be arrays and the same length."
    with pytest.raises(
        ValueError,
        match=re.escape(message),
    ):
        snappy.SurfaceEdgeRefinement(spacing=[2 * u.mm], entities=[Surface(name="test")])

    snappy.SurfaceEdgeRefinement(
        spacing=[2 * u.mm], distances=[5 * u.mm], entities=[Surface(name="test")]
    )

    snappy.SurfaceEdgeRefinement(entities=[Surface(name="test")])

    snappy.SurfaceEdgeRefinement(spacing=2 * u.mm, entities=[Surface(name="test")])

    snappy.SurfaceMeshingParams(
        defaults=snappy.SurfaceMeshingDefaults(
            min_spacing=3 * u.mm, max_spacing=6 * u.mm, gap_resolution=0.1 * u.mm
        ),
        refinements=[
            snappy.SurfaceEdgeRefinement(
                spacing=[2 * u.mm], distances=[5 * u.mm], entities=[Surface(name="test")]
            ),
            snappy.SurfaceEdgeRefinement(spacing=2 * u.mm, entities=[Surface(name="test")]),
            snappy.SurfaceEdgeRefinement(entities=[Surface(name="test")]),
        ],
    )


def test_snappy_edge_refinement_increasing_values_validator():
    message = "Spacings and distances must be increasing arrays."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.SurfaceEdgeRefinement(
            spacing=[2 * u.mm, 1 * u.mm],
            distances=[5 * u.mm, 6 * u.mm],
            entities=[Surface(name="test")],
        )

    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.SurfaceEdgeRefinement(
            spacing=[2 * u.mm, 3 * u.mm],
            distances=[5 * u.mm, 4 * u.mm],
            entities=[Surface(name="test")],
        )

    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.SurfaceEdgeRefinement(
            spacing=[2 * u.mm, 1 * u.mm],
            distances=[5 * u.mm, 4 * u.mm],
            entities=[Surface(name="test")],
        )


def test_snappy_body_refinement_validator():
    message = "No refinement (gap_resolution, min_spacing, max_spacing, proximity_spacing) specified in `BodyRefinement`."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.BodyRefinement(bodies=SnappyBody(name="body1", surfaces=[Surface(name="surface")]))

    snappy.BodyRefinement(
        bodies=SnappyBody(name="body1", surfaces=[Surface(name="surface")]), min_spacing=2 * u.mm
    )

    snappy.BodyRefinement(
        bodies=SnappyBody(name="body1", surfaces=[Surface(name="surface")]), max_spacing=2 * u.mm
    )

    snappy.BodyRefinement(
        bodies=SnappyBody(name="body1", surfaces=[Surface(name="surface")]),
        proximity_spacing=2 * u.mm,
    )

    snappy.BodyRefinement(
        bodies=SnappyBody(name="body1", surfaces=[Surface(name="surface")]), gap_resolution=2 * u.mm
    )
