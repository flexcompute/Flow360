import re

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.meshing_specs import (
    VolumeMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    ModularMeshingWorkflow,
    VolumeMeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    UniformRefinement,
)
from flow360.component.simulation.primitives import Box, Cylinder, SnappyBody, Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
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


def _make_snappy_params_with_volume_uniform_refinement(refinement):
    """Helper to build SimulationParams with a UniformRefinement in volume meshing."""
    with SI_unit_system:
        return SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1 * u.mm,
                        max_spacing=10 * u.mm,
                        gap_resolution=0.1 * u.mm,
                    )
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(
                        boundary_layer_first_layer_thickness=1 * u.mm,
                    ),
                    refinements=[refinement],
                ),
                zones=[AutomatedFarfield()],
            )
        )


def test_volume_uniform_refinement_rotated_box_project_to_surface():
    """
    A UniformRefinement with a rotated Box placed in volume meshing with
    project_to_surface=True must trigger the same snappy validation error
    as if it were placed directly in the surface meshing refinements.
    """
    rotated_box = Box(
        center=[0, 0, 0] * u.m,
        size=[1, 1, 1] * u.m,
        axis_of_rotation=[0, 0, 1],
        angle_of_rotation=45 * u.deg,
        name="rotated_box",
    )

    refinement = UniformRefinement(
        spacing=5 * u.mm,
        entities=[rotated_box],
        project_to_surface=True,
    )

    params = _make_snappy_params_with_volume_uniform_refinement(refinement)

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is not None, (
        "Expected validation error for rotated Box in volume UniformRefinement "
        "with project_to_surface=True"
    )
    error_messages = [e["msg"] for e in errors]
    assert any("angle_of_rotation" in msg or "axes aligned" in msg for msg in error_messages)


def test_volume_uniform_refinement_hollow_cylinder_project_to_surface():
    """
    A UniformRefinement with a hollow Cylinder (inner_radius > 0) placed in
    volume meshing with project_to_surface=True must trigger the same snappy
    validation error as if it were in the surface meshing refinements.
    """
    hollow_cylinder = Cylinder(
        name="hollow_cyl",
        inner_radius=3 * u.mm,
        outer_radius=7 * u.mm,
        axis=[0, 0, 1],
        center=[0, 0, 0] * u.m,
        height=10 * u.mm,
    )

    refinement = UniformRefinement(
        spacing=5 * u.mm,
        entities=[hollow_cylinder],
        project_to_surface=True,
    )

    params = _make_snappy_params_with_volume_uniform_refinement(refinement)

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is not None, (
        "Expected validation error for hollow Cylinder in volume UniformRefinement "
        "with project_to_surface=True"
    )
    error_messages = [e["msg"] for e in errors]
    assert any("inner_radius" in msg or "full cylinders" in msg for msg in error_messages)


def test_volume_uniform_refinement_default_project_to_surface():
    """
    When project_to_surface is None (the default, which acts as True for snappy),
    the same snappy constraints should be enforced.
    """
    rotated_box = Box(
        center=[0, 0, 0] * u.m,
        size=[1, 1, 1] * u.m,
        axis_of_rotation=[0, 0, 1],
        angle_of_rotation=90 * u.deg,
        name="rotated_box_default",
    )

    refinement = UniformRefinement(
        spacing=5 * u.mm,
        entities=[rotated_box],
    )

    params = _make_snappy_params_with_volume_uniform_refinement(refinement)

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is not None, (
        "Expected validation error for rotated Box in volume UniformRefinement "
        "with default project_to_surface (None)"
    )
    error_messages = [e["msg"] for e in errors]
    assert any("angle_of_rotation" in msg or "axes aligned" in msg for msg in error_messages)


def test_volume_uniform_refinement_project_to_surface_false_skips_validation():
    """
    When project_to_surface=False, snappy-specific constraints on entities
    should NOT be enforced since the refinement won't be projected to the
    surface mesh.
    """
    rotated_box = Box(
        center=[0, 0, 0] * u.m,
        size=[1, 1, 1] * u.m,
        axis_of_rotation=[0, 0, 1],
        angle_of_rotation=45 * u.deg,
        name="rotated_box_no_project",
    )

    refinement = UniformRefinement(
        spacing=5 * u.mm,
        entities=[rotated_box],
        project_to_surface=False,
    )

    params = _make_snappy_params_with_volume_uniform_refinement(refinement)

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is None, "No snappy validation error expected when project_to_surface=False"
