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
    with mock_validation_context, SI_unit_system, pytest.raises(
        ValueError, match=re.escape(message)
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
    with mock_validation_context, SI_unit_system, pytest.raises(
        ValueError, match=re.escape(message)
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
            snappy.SurfaceEdgeRefinement(spacing=2 * u.mm, entities=[Surface(name="test2")]),
            snappy.SurfaceEdgeRefinement(entities=[Surface(name="test3")]),
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


def test_snappy_proximity_spacing_clamped_to_default_min_spacing():
    """When min_spacing is not set on a BodyRefinement but proximity_spacing exceeds
    defaults.min_spacing, proximity_spacing should be clamped to defaults.min_spacing."""
    with SI_unit_system:
        body = SnappyBody(name="body1", surfaces=[Surface(name="surface")])
        params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=10 * u.mm, gap_resolution=0.1 * u.mm
            ),
            refinements=[
                snappy.BodyRefinement(
                    bodies=body,
                    proximity_spacing=5 * u.mm,
                    max_spacing=10 * u.mm,
                ),
            ],
        )
        # proximity_spacing (5 mm) > defaults.min_spacing (3 mm) => clamped to 3 mm
        assert params.refinements[0].proximity_spacing == 3 * u.mm


def test_snappy_proximity_spacing_not_clamped_when_below_default_min_spacing():
    """When proximity_spacing is already <= defaults.min_spacing, it should remain unchanged."""
    with SI_unit_system:
        body = SnappyBody(name="body1", surfaces=[Surface(name="surface")])
        params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=10 * u.mm, gap_resolution=0.1 * u.mm
            ),
            refinements=[
                snappy.BodyRefinement(
                    bodies=body,
                    proximity_spacing=2 * u.mm,
                    max_spacing=10 * u.mm,
                ),
            ],
        )
        # proximity_spacing (2 mm) <= defaults.min_spacing (3 mm) => unchanged
        assert params.refinements[0].proximity_spacing == 2 * u.mm


def test_snappy_proximity_spacing_with_explicit_min_spacing_uses_entity_validator():
    """When both min_spacing and proximity_spacing are set on the refinement,
    the entity-level validator clamps proximity_spacing to min_spacing (not defaults)."""
    with SI_unit_system:
        body = SnappyBody(name="body1", surfaces=[Surface(name="surface")])
        refinement = snappy.BodyRefinement(
            bodies=body,
            min_spacing=4 * u.mm,
            proximity_spacing=6 * u.mm,
            max_spacing=10 * u.mm,
        )
        # Entity-level validator: proximity_spacing (6 mm) > min_spacing (4 mm) => clamped to 4 mm
        assert refinement.proximity_spacing == 4 * u.mm


def test_snappy_proximity_spacing_not_clamped_when_min_spacing_is_set():
    """When min_spacing is explicitly set, the general validator should not interfere."""
    with SI_unit_system:
        body = SnappyBody(name="body1", surfaces=[Surface(name="surface")])
        params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=10 * u.mm, gap_resolution=0.1 * u.mm
            ),
            refinements=[
                snappy.BodyRefinement(
                    bodies=body,
                    min_spacing=4 * u.mm,
                    proximity_spacing=3 * u.mm,
                    max_spacing=10 * u.mm,
                ),
            ],
        )
        # min_spacing is set => general validator skipped, entity validator sees
        # proximity_spacing (3 mm) < min_spacing (4 mm) => no clamping
        assert params.refinements[0].proximity_spacing == 3 * u.mm
