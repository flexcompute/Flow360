import pydantic as pd
import pytest

from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationVolume,
    UniformRefinement,
)
from flow360.component.simulation.primitives import AxisymmetricBody, Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import CGS_unit_system
from flow360.component.simulation.validation.validation_context import (
    VOLUME_MESH,
    ParamsValidationInfo,
    ValidationContext,
)

non_beta_mesher_context = ParamsValidationInfo({}, [])
non_beta_mesher_context.is_beta_mesher = False

beta_mesher_context = ParamsValidationInfo({}, [])
beta_mesher_context.is_beta_mesher = True


def test_disable_invalid_axisymmetric_body_construction():
    import re

    with pytest.raises(
        pd.ValidationError,
        match=re.escape(
            "Expect profile samples to be (Axial, Radial) samples with positive Radial. Found invalid point: [-1.  1.  3.] cm."
        ),
    ):
        with CGS_unit_system:
            cylinder_1 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 0), (-1, 1, 3), (1, 1), (1, 0)],
            )

    with pytest.raises(
        pd.ValidationError,
        match=re.escape(
            "Expect first profile sample to be (Axial, 0.0). Found invalid point: [-1.  1.] cm."
        ),
    ):
        with CGS_unit_system:
            cylinder_1 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 1), (1, 2)],
            )

    with pytest.raises(
        pd.ValidationError,
        match=re.escape(
            "Expect last profile sample to be (Axial, 0.0). Found invalid point: [1. 1.] cm."
        ),
    ):
        with CGS_unit_system:
            cylinder_1 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 0), (-1, 1), (1, 1)],
            )


def test_disable_multiple_cylinder_in_one_ratataion_cylinder():
    with pytest.raises(
        pd.ValidationError,
        match="Only single instance is allowed in entities for each RotationCylinder.",
        match="Only single instance is allowed in entities for each `RotationVolume`.",
    ):
        with CGS_unit_system:
            cylinder_1 = Cylinder(
                name="1",
                outer_radius=12,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            cylinder_2 = Cylinder(
                name="2",
                outer_radius=2,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        RotationVolume(
                            entities=[cylinder_1, cylinder_2],
                            spacing_axial=20,
                            spacing_radial=0.2,
                            spacing_circumferential=20,
                            enclosed_entities=[
                                Surface(name="hub"),
                            ],
                        ),
                        AutomatedFarfield(),
                    ],
                )
            )


def test_limit_cylinder_entity_name_length_in_rotation_volume():
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"The name \(very_long_cylinder_name\) of `Cylinder` entity in `RotationVolume`"
        + " exceeds 18 characters limit.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="very_long_cylinder_name",
                    outer_radius=12,
                    height=2,
                    axis=(0, 1, 0),
                    center=(0, 5, 0),
                )
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=20,
                    spacing_radial=0.2,
                    spacing_circumferential=20,
                    enclosed_entities=[
                        Surface(name="hub"),
                    ],
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder2 = Cylinder(
                name="very_long_cylinder_name",
                outer_radius=12,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            _ = RotationVolume(
                entities=[cylinder2],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[
                    Surface(name="hub"),
                ],
            )


def test_limit_axisymmetric_body_in_rotation_volume():
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`AxisymmetricBody` entity for `RotationVolume` is only supported with the beta mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                cylinder_1 = AxisymmetricBody(
                    name="1",
                    axis=(0, 0, 1),
                    center=(0, 5, 0),
                    profile_curve=[(-1, 0), (-1, 1), (1, 1), (1, 0)],
                )

                _ = RotationVolume(
                    entities=[cylinder_1],
                    spacing_axial=20,
                    spacing_radial=0.2,
                    spacing_circumferential=20,
                    enclosed_entities=[
                        Surface(name="hub"),
                    ],
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder_2 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 0), (-1, 1), (1, 1), (1, 0)],
            )

            _ = RotationVolume(
                entities=[cylinder_2],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[
                    Surface(name="hub"),
                ],
            )


def test_reuse_of_same_cylinder():
    with pytest.raises(
        pd.ValidationError,
        match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `RotationVolume` at the same time is not allowed.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        RotationVolume(
                            entities=[cylinder],
                            spacing_axial=20,
                            spacing_radial=0.2,
                            spacing_circumferential=20,
                            enclosed_entities=[
                                Surface(name="hub"),
                            ],
                        ),
                        AutomatedFarfield(),
                    ],
                    refinements=[
                        AxisymmetricRefinement(
                            entities=[cylinder],
                            spacing_axial=0.1,
                            spacing_radial=0.2,
                            spacing_circumferential=0.3,
                        )
                    ],
                )
            )

    with CGS_unit_system:
        cylinder = Cylinder(
            name="Okay to reuse",
            outer_radius=1,
            height=12,
            axis=(0, 1, 0),
            center=(0, 5, 0),
        )
        SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    RotationVolume(
                        entities=[cylinder],
                        spacing_axial=20,
                        spacing_radial=0.2,
                        spacing_circumferential=20,
                        enclosed_entities=[
                            Surface(name="hub"),
                        ],
                    ),
                    AutomatedFarfield(),
                ],
                refinements=[
                    UniformRefinement(
                        entities=[cylinder],
                        spacing=0.1,
                    )
                ],
            )
        )

    with pytest.raises(
        pd.ValidationError,
        match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    refinements=[
                        UniformRefinement(entities=[cylinder], spacing=0.1),
                        AxisymmetricRefinement(
                            entities=[cylinder],
                            spacing_axial=0.1,
                            spacing_radial=0.1,
                            spacing_circumferential=0.1,
                        ),
                    ],
                )
            )

    with pytest.raises(
        pd.ValidationError,
        match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    refinements=[
                        UniformRefinement(entities=[cylinder], spacing=0.1),
                        UniformRefinement(entities=[cylinder], spacing=0.2),
                    ],
                )
            )
