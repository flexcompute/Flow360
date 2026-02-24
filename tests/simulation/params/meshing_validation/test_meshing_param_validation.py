import re

import pydantic as pd
import pytest

from flow360 import u
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.face_params import (
    GeometryRefinement,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.meshing_specs import (
    MeshingDefaults,
    OctreeSpacing,
    VolumeMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
    VolumeMeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    CustomZones,
    FullyMovingFloor,
    RotationVolume,
    StaticFloor,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
    WheelBelts,
    WindTunnelFarfield,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    SeedpointVolume,
    SnappyBody,
    Sphere,
    Surface,
)
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import CGS_unit_system, SI_unit_system
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ParamsValidationInfo,
    ValidationContext,
)

non_beta_mesher_context = ParamsValidationInfo({}, [])
non_beta_mesher_context.is_beta_mesher = False

non_gai_context = ParamsValidationInfo({}, [])
non_gai_context.use_geometry_AI = False

beta_mesher_context = ParamsValidationInfo({}, [])
beta_mesher_context.is_beta_mesher = True
beta_mesher_context.project_length_unit = "mm"


def test_structured_box_only_in_beta_mesher():
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`StructuredBoxRefinement` is only supported with the beta mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                porous_medium = Box.from_principal_axes(
                    name="porousRegion",
                    center=(0, 1, 1),
                    size=(1, 2, 1),
                    axes=((2, 2, 0), (-2, 2, 0)),
                )
                _ = StructuredBoxRefinement(
                    entities=[porous_medium],
                    spacing_axis1=10,
                    spacing_axis2=10,
                    spacing_normal=10,
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            porous_medium = Box.from_principal_axes(
                name="porousRegion",
                center=(0, 1, 1),
                size=(1, 2, 1),
                axes=((2, 2, 0), (-2, 2, 0)),
            )
            _ = StructuredBoxRefinement(
                entities=[porous_medium],
                spacing_axis1=10,
                spacing_axis2=10,
                spacing_normal=10,
            )


def test_no_reuse_box_in_refinements():
    with pytest.raises(
        pd.ValidationError,
        match=r"Using Volume entity `box-reused` in `StructuredBoxRefinement`, `UniformRefinement` at the same time is not allowed.",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                porous_medium = Box.from_principal_axes(
                    name="box-reused",
                    center=(0, 1, 1),
                    size=(1, 2, 1),
                    axes=((2, 2, 0), (-2, 2, 0)),
                )
                structured_box_refine = StructuredBoxRefinement(
                    entities=[porous_medium],
                    spacing_axis1=10,
                    spacing_axis2=10,
                    spacing_normal=10,
                )
                uniform_refine = UniformRefinement(entities=[porous_medium], spacing=10)

                SimulationParams(
                    meshing=MeshingParams(
                        refinements=[uniform_refine, structured_box_refine],
                    )
                )


def test_disable_invalid_axisymmetric_body_construction():
    import re

    with pytest.raises(
        pd.ValidationError,
        match=re.escape("Value error, arg '(-1, 1, 3)' needs to be a collection of 2 values"),
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
        match=re.escape("should have at least 2 items"),
    ):
        with CGS_unit_system:
            AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[],
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

    with pytest.raises(
        pd.ValidationError,
        match=re.escape("Profile curve has duplicate consecutive points at indices 1 and 2"),
    ):
        with CGS_unit_system:
            invalid = AxisymmetricBody(
                name="1",
                axis=(1, 0, 0),
                center=(0, 3, 0),
                profile_curve=[(-1, 0), (-1, 1.23), (-1, 1.23), (1, 1), (1, 0)],
            )


def test_disable_multiple_cylinder_in_one_rotation_volume(mock_validation_context):
    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match="Only single instance is allowed in entities for each `RotationVolume`.",
        ),
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
    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match="Only single instance is allowed in entities for each `RotationVolume`.",
        ),
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
                meshing=ModularMeshingWorkflow(
                    zones=[
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


def test_sphere_in_rotation_volume_only_in_beta_mesher():
    """Test that Sphere entity for RotationVolume is only supported with the beta mesher."""
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`Sphere` entity for `RotationVolume` is only supported with the beta mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                sphere = Sphere(
                    name="rotation_sphere",
                    center=(0, 0, 0),
                    radius=10,
                )
                _ = RotationVolume(
                    entities=[sphere],
                    spacing_circumferential=0.5,
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            sphere = Sphere(
                name="rotation_sphere",
                center=(0, 0, 0),
                radius=10,
            )
            _ = RotationVolume(
                entities=[sphere],
                spacing_circumferential=0.5,
            )


def test_sphere_rotation_volume_spacing_requirements():
    """Test spacing requirements for Sphere vs Cylinder/AxisymmetricBody in RotationVolume."""
    # Test 1: Sphere without spacing_circumferential should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"`spacing_circumferential` is required for `Sphere` entities",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                sphere = Sphere(name="sphere", center=(0, 0, 0), radius=10)
                _ = RotationVolume(
                    entities=[sphere],
                )

    # Test 2: Sphere with spacing_axial should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"`spacing_axial` must not be specified for `Sphere` entities",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                sphere = Sphere(name="sphere", center=(0, 0, 0), radius=10)
                _ = RotationVolume(
                    entities=[sphere],
                    spacing_circumferential=0.5,
                    spacing_axial=0.5,
                )

    # Test 3: Sphere with spacing_radial should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"`spacing_radial` must not be specified for `Sphere` entities",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                sphere = Sphere(name="sphere", center=(0, 0, 0), radius=10)
                _ = RotationVolume(
                    entities=[sphere],
                    spacing_circumferential=0.5,
                    spacing_radial=0.5,
                )

    # Test 4: Cylinder without spacing_axial should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"`spacing_axial` is required for `Cylinder` or `AxisymmetricBody` entities",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cyl",
                    center=(0, 0, 0),
                    axis=(0, 0, 1),
                    height=10,
                    outer_radius=5,
                )
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_circumferential=0.5,
                    spacing_radial=0.5,
                )

    # Test 5: Cylinder without spacing_radial should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"`spacing_radial` is required for `Cylinder` or `AxisymmetricBody` entities",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cyl",
                    center=(0, 0, 0),
                    axis=(0, 0, 1),
                    height=10,
                    outer_radius=5,
                )
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_circumferential=0.5,
                    spacing_axial=0.5,
                )

    # Test 6: Cylinder without spacing_circumferential should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"`spacing_circumferential` is required for `Cylinder` or `AxisymmetricBody`",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cyl",
                    center=(0, 0, 0),
                    axis=(0, 0, 1),
                    height=10,
                    outer_radius=5,
                )
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=0.5,
                    spacing_radial=0.5,
                )


def test_sphere_rotation_volume_with_enclosed_entities():
    """Test that Sphere RotationVolume supports enclosed_entities."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            sphere = Sphere(name="outer_sphere", center=(0, 0, 0), radius=10)
            inner_sphere = Sphere(name="inner_sphere", center=(0, 0, 0), radius=5)
            _ = RotationVolume(
                entities=[sphere],
                spacing_circumferential=0.5,
                enclosed_entities=[inner_sphere, Surface(name="hub")],
            )


def test_sphere_in_enclosed_entities_only_in_beta_mesher():
    """Test that Sphere in enclosed_entities is only supported with the beta mesher."""
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`Sphere` entity in `RotationVolume.enclosed_entities` is only supported with the beta mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="outer_cyl",
                    center=(0, 0, 0),
                    axis=(0, 0, 1),
                    height=10,
                    outer_radius=5,
                )
                inner_sphere = Sphere(name="inner_sphere", center=(0, 0, 0), radius=2)
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=0.5,
                    spacing_radial=0.5,
                    spacing_circumferential=0.5,
                    enclosed_entities=[inner_sphere],
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="outer_cyl",
                center=(0, 0, 0),
                axis=(0, 0, 1),
                height=10,
                outer_radius=5,
            )
            inner_sphere = Sphere(name="inner_sphere", center=(0, 0, 0), radius=2)
            _ = RotationVolume(
                entities=[cylinder],
                spacing_axial=0.5,
                spacing_radial=0.5,
                spacing_circumferential=0.5,
                enclosed_entities=[inner_sphere],
            )


def test_reuse_of_same_cylinder(mock_validation_context):
    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `RotationVolume` at the same time is not allowed.",
        ),
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

    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `RotationVolume` at the same time is not allowed.",
        ),
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
                meshing=ModularMeshingWorkflow(
                    volume_meshing=VolumeMeshingParams(
                        refinements=[
                            AxisymmetricRefinement(
                                entities=[cylinder],
                                spacing_axial=0.1,
                                spacing_radial=0.2,
                                spacing_circumferential=0.3,
                            )
                        ],
                        defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1),
                    ),
                    zones=[
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

    with CGS_unit_system:
        cylinder = Cylinder(
            name="Okay to reuse",
            outer_radius=1,
            height=12,
            axis=(0, 1, 0),
            center=(0, 5, 0),
        )
        SimulationParams(
            meshing=ModularMeshingWorkflow(
                volume_meshing=VolumeMeshingParams(
                    refinements=[
                        UniformRefinement(
                            entities=[cylinder],
                            spacing=0.1,
                        )
                    ],
                    defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1),
                ),
                zones=[
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
            )
        )

    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
        ),
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

    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
        ),
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
                meshing=ModularMeshingWorkflow(
                    volume_meshing=VolumeMeshingParams(
                        refinements=[
                            UniformRefinement(entities=[cylinder], spacing=0.1),
                            AxisymmetricRefinement(
                                entities=[cylinder],
                                spacing_axial=0.1,
                                spacing_radial=0.1,
                                spacing_circumferential=0.1,
                            ),
                        ],
                        defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1),
                    ),
                    zones=[AutomatedFarfield()],
                )
            )

    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
        ),
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

    with (
        mock_validation_context,
        pytest.raises(
            pd.ValidationError,
            match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
        ),
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
                meshing=ModularMeshingWorkflow(
                    volume_meshing=VolumeMeshingParams(
                        refinements=[
                            UniformRefinement(entities=[cylinder], spacing=0.1),
                            UniformRefinement(entities=[cylinder], spacing=0.2),
                        ],
                        defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1),
                    ),
                    zones=[AutomatedFarfield()],
                )
            )


def test_axisymmetric_body_in_uniform_refinement():
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            axisymmetric_body = AxisymmetricBody(
                name="a",
                axis=(0, 0, 1),
                center=(0, 0, 0),
                profile_curve=[(-2, 0), (-2, 1), (2, 1.5), (2, 0)],
            )
            MeshingParams(
                refinements=[
                    UniformRefinement(
                        entities=[axisymmetric_body],
                        spacing=0.1,
                    )
                ],
            )

    # raises without beta mesher
    with pytest.raises(
        pd.ValidationError,
        match=r"`AxisymmetricBody` entity for `UniformRefinement` is supported only with beta mesher",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                axisymmetric_body = AxisymmetricBody(
                    name="1",
                    axis=(0, 0, 1),
                    center=(0, 0, 0),
                    profile_curve=[(-1, 0), (-1, 1), (1, 1), (1, 0)],
                )
                UniformRefinement(
                    entities=[axisymmetric_body],
                    spacing=0.1,
                )


def test_require_mesh_zones():
    with SI_unit_system:
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm,
                    max_spacing=5 * u.mm,
                    gap_resolution=0.001 * u.mm,
                ),
            ),
            zones=[AutomatedFarfield()],
        )

    with SI_unit_system:
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm,
                    max_spacing=5 * u.mm,
                    gap_resolution=0.01 * u.mm,
                ),
            ),
            zones=[
                CustomZones(
                    name="custom_zones",
                    entities=[SeedpointVolume(name="fluid", point_in_mesh=(0, 0, 0) * u.mm)],
                )
            ],
        )

    message = "snappyHexMeshing requires at least one `SeedpointVolume` when not using `AutomatedFarfield`."
    with pytest.raises(
        ValueError,
        match=re.escape(message),
    ):
        with SI_unit_system:
            ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1 * u.mm,
                        max_spacing=5 * u.mm,
                        gap_resolution=0.01 * u.mm,
                    )
                ),
                zones=[UserDefinedFarfield()],
            )


def test_bad_refinements():
    message = "Default maximum spacing (5.0 mm) is lower than refinement minimum spacing (6.0 mm) and maximum spacing is not provided for BodyRefinement."
    with pytest.raises(
        ValueError,
        match=re.escape(message),
    ):
        snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
            ),
            refinements=[
                snappy.BodyRefinement(
                    min_spacing=6 * u.mm, bodies=[SnappyBody(name="bbb", surfaces=[])]
                )
            ],
        )

    message = "Default minimum spacing (1.0 mm) is higher than refinement maximum spacing (0.5 mm) and minimum spacing is not provided for BodyRefinement."
    with pytest.raises(
        ValueError,
        match=re.escape(message),
    ):
        snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
            ),
            refinements=[
                snappy.BodyRefinement(
                    max_spacing=0.5 * u.mm, bodies=[SnappyBody(name="bbb", surfaces=[])]
                )
            ],
        )


def test_duplicate_refinement_type_per_entity():
    """Raise when the same refinement type is applied twice to one entity."""
    body = SnappyBody(name="car_body", surfaces=[])
    surface = Surface(name="wing")
    defaults = snappy.SurfaceMeshingDefaults(
        min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
    )

    # -- Two BodyRefinements targeting the same SnappyBody --
    with pytest.raises(
        pd.ValidationError,
        match=r"`BodyRefinement` is applied 2 times to entity `car_body`",
    ):
        snappy.SurfaceMeshingParams(
            defaults=defaults,
            refinements=[
                snappy.BodyRefinement(min_spacing=2 * u.mm, bodies=[body]),
                snappy.BodyRefinement(max_spacing=4 * u.mm, bodies=[body]),
            ],
        )

    # -- Two RegionRefinements targeting the same Surface --
    with pytest.raises(
        pd.ValidationError,
        match=r"`RegionRefinement` is applied 2 times to entity `wing`",
    ):
        snappy.SurfaceMeshingParams(
            defaults=defaults,
            refinements=[
                snappy.RegionRefinement(
                    min_spacing=1 * u.mm, max_spacing=3 * u.mm, regions=[surface]
                ),
                snappy.RegionRefinement(
                    min_spacing=2 * u.mm, max_spacing=4 * u.mm, regions=[surface]
                ),
            ],
        )

    # -- Two SurfaceEdgeRefinements targeting the same SnappyBody --
    with pytest.raises(
        pd.ValidationError,
        match=r"`SurfaceEdgeRefinement` is applied 2 times to entity `car_body`",
    ):
        snappy.SurfaceMeshingParams(
            defaults=defaults,
            refinements=[
                snappy.SurfaceEdgeRefinement(spacing=0.5 * u.mm, entities=[body]),
                snappy.SurfaceEdgeRefinement(spacing=1 * u.mm, entities=[body]),
            ],
        )

    # -- Two SurfaceEdgeRefinements targeting the same Surface --
    with pytest.raises(
        pd.ValidationError,
        match=r"`SurfaceEdgeRefinement` is applied 2 times to entity `wing`",
    ):
        snappy.SurfaceMeshingParams(
            defaults=defaults,
            refinements=[
                snappy.SurfaceEdgeRefinement(spacing=0.5 * u.mm, entities=[surface]),
                snappy.SurfaceEdgeRefinement(spacing=1 * u.mm, entities=[surface]),
            ],
        )


def test_duplicate_refinement_different_types_is_allowed():
    """Different refinement types on the same entity should NOT raise."""
    body = SnappyBody(name="car_body", surfaces=[])
    surface = Surface(name="wing")
    defaults = snappy.SurfaceMeshingDefaults(
        min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
    )

    # BodyRefinement + SurfaceEdgeRefinement on the same SnappyBody is fine
    snappy.SurfaceMeshingParams(
        defaults=defaults,
        refinements=[
            snappy.BodyRefinement(min_spacing=2 * u.mm, bodies=[body]),
            snappy.SurfaceEdgeRefinement(spacing=0.5 * u.mm, entities=[body]),
        ],
    )

    # RegionRefinement + SurfaceEdgeRefinement on the same Surface is fine
    snappy.SurfaceMeshingParams(
        defaults=defaults,
        refinements=[
            snappy.RegionRefinement(min_spacing=1 * u.mm, max_spacing=3 * u.mm, regions=[surface]),
            snappy.SurfaceEdgeRefinement(spacing=0.5 * u.mm, entities=[surface]),
        ],
    )


def test_duplicate_refinement_different_entities_is_allowed():
    """Same refinement type on different entities should NOT raise."""
    body1 = SnappyBody(name="car_body", surfaces=[])
    body2 = SnappyBody(name="other_body", surfaces=[])
    defaults = snappy.SurfaceMeshingDefaults(
        min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
    )

    snappy.SurfaceMeshingParams(
        defaults=defaults,
        refinements=[
            snappy.BodyRefinement(min_spacing=2 * u.mm, bodies=[body1]),
            snappy.BodyRefinement(min_spacing=3 * u.mm, bodies=[body2]),
        ],
    )


def test_duplicate_refinement_body_and_surface_same_name_is_allowed():
    """SurfaceEdgeRefinement on a SnappyBody and a Surface sharing a name should NOT raise."""
    body = SnappyBody(name="shared_name", surfaces=[])
    surface = Surface(name="shared_name")
    defaults = snappy.SurfaceMeshingDefaults(
        min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
    )

    snappy.SurfaceMeshingParams(
        defaults=defaults,
        refinements=[
            snappy.SurfaceEdgeRefinement(spacing=0.5 * u.mm, entities=[body]),
            snappy.SurfaceEdgeRefinement(spacing=1 * u.mm, entities=[surface]),
        ],
    )


def test_box_entity_enclosed_only_in_beta_mesher():
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`Box` entity in `RotationVolume.enclosed_entities` is only supported with the beta mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cylinder",
                    outer_radius=1,
                    height=12,
                    axis=(0, 1, 0),
                    center=(0, 5, 0),
                )
                box_entity = Box.from_principal_axes(
                    name="box",
                    center=(0, 1, 1),
                    size=(1, 2, 1),
                    axes=((2, 2, 0), (-2, 2, 0)),
                )
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=20,
                    spacing_radial=0.2,
                    spacing_circumferential=20,
                    enclosed_entities=[box_entity],
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cylinder",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            box_entity = Box.from_principal_axes(
                name="box",
                center=(0, 1, 1),
                size=(1, 2, 1),
                axes=((2, 2, 0), (-2, 2, 0)),
            )
            _ = RotationVolume(
                entities=[cylinder],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[box_entity],
            )


def test_octree_spacing():
    spacing = OctreeSpacing(base_spacing=2 * u.mm)

    assert spacing[0] == 2 * u.mm
    assert spacing[3] == 2 * u.mm * (2**-3)
    assert spacing[-4] == 2 * u.mm * (2**4)
    assert spacing[1] == 2 * u.mm * (2**-1)

    with pytest.raises(pd.ValidationError):
        _ = spacing[0.2]

    assert spacing.to_level(2 * u.mm) == (0, True)
    assert spacing.to_level(4 * u.mm) == (-1, True)
    assert spacing.to_level(0.5 * u.mm) == (2, True)
    assert spacing.to_level(3.9993 * u.mm) == (0, False)
    assert spacing.to_level(3.9999999999993 * u.mm) == (-1, True)


def test_set_default_octree_spacing():
    surface_meshing = snappy.SurfaceMeshingParams(
        defaults=snappy.SurfaceMeshingDefaults(
            min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
        )
    )

    assert surface_meshing.octree_spacing is None

    with ValidationContext(SURFACE_MESH, beta_mesher_context):
        surface_meshing = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
            )
        )

    assert surface_meshing.octree_spacing.base_spacing == 1 * u.mm
    assert surface_meshing.octree_spacing[2] == 0.25 * u.mm
    assert surface_meshing.octree_spacing.to_level(2 * u.mm) == (-1, True)


def test_set_spacing_with_value():
    surface_meshing = snappy.SurfaceMeshingParams(
        defaults=snappy.SurfaceMeshingDefaults(
            min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
        ),
        octree_spacing=3 * u.mm,
    )

    assert surface_meshing.octree_spacing.base_spacing == 3 * u.mm

    with pytest.raises(pd.ValidationError):
        surface_meshing = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
            ),
            octree_spacing=-3 * u.mm,
        )


def test_set_spacing_with_base_spacing_alias():
    """Test that base_spacing alias still works for backward compatibility."""
    surface_meshing = snappy.SurfaceMeshingParams(
        defaults=snappy.SurfaceMeshingDefaults(
            min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
        ),
        base_spacing=3 * u.mm,
    )

    assert surface_meshing.octree_spacing.base_spacing == 3 * u.mm


def test_quasi_3d_periodic_only_in_legacy_mesher():
    # raises when legacy mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"Only legacy mesher can support quasi-3d-periodic",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            my_farfield = AutomatedFarfield(method="quasi-3d-periodic")

    # does not raise with legacy mesher on
    with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
        my_farfield = AutomatedFarfield(method="quasi-3d-periodic")


def test_enforced_half_model_only_in_beta_mesher():
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`domain_type` is only supported when using both GAI surface mesher and beta volume mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            AutomatedFarfield(domain_type="half_body_positive_y")

    # raise when GAI is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`domain_type` is only supported when using both GAI surface mesher and beta volume mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_gai_context):
            AutomatedFarfield(domain_type="full_body")


def test_enclosed_entities_none_does_not_raise():
    with CGS_unit_system:
        cylinder = Cylinder(
            name="cylinder",
            outer_radius=1,
            height=12,
            axis=(0, 1, 0),
            center=(0, 5, 0),
        )
        # Should not raise even when enclosed_entities is explicitly None
        _ = RotationVolume(
            entities=[cylinder],
            spacing_axial=20,
            spacing_radial=0.2,
            spacing_circumferential=20,
        )


def test_stationary_enclosed_entities_only_in_beta_mesher():
    """Test that stationary_enclosed_entities is only supported with beta mesher."""
    # raises when beta mesher is off
    with pytest.raises(
        pd.ValidationError,
        match=r"`stationary_enclosed_entities` in `RotationVolume` is only supported with the beta mesher.",
    ):
        with ValidationContext(VOLUME_MESH, non_beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cylinder",
                    outer_radius=1,
                    height=12,
                    axis=(0, 1, 0),
                    center=(0, 5, 0),
                )
                surface1 = Surface(name="hub")
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=20,
                    spacing_radial=0.2,
                    spacing_circumferential=20,
                    enclosed_entities=[surface1],
                    stationary_enclosed_entities=[surface1],
                )

    # does not raise with beta mesher on
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cylinder",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            surface1 = Surface(name="hub")
            _ = RotationVolume(
                entities=[cylinder],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[surface1],
                stationary_enclosed_entities=[surface1],
            )


def test_stationary_enclosed_entities_requires_enclosed_entities():
    """Test that stationary_enclosed_entities cannot be specified when enclosed_entities is None."""
    with pytest.raises(
        pd.ValidationError,
        match=r"`stationary_enclosed_entities` cannot be specified when `enclosed_entities` is None.",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cylinder",
                    outer_radius=1,
                    height=12,
                    axis=(0, 1, 0),
                    center=(0, 5, 0),
                )
                surface1 = Surface(name="hub")
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=20,
                    spacing_radial=0.2,
                    spacing_circumferential=20,
                    enclosed_entities=None,
                    stationary_enclosed_entities=[surface1],
                )


def test_stationary_enclosed_entities_must_be_subset():
    """Test that stationary_enclosed_entities must be a subset of enclosed_entities."""
    with pytest.raises(
        pd.ValidationError,
        match=r"All entities in `stationary_enclosed_entities` must be present in `enclosed_entities`.",
    ):
        with ValidationContext(VOLUME_MESH, beta_mesher_context):
            with CGS_unit_system:
                cylinder = Cylinder(
                    name="cylinder",
                    outer_radius=1,
                    height=12,
                    axis=(0, 1, 0),
                    center=(0, 5, 0),
                )
                surface1 = Surface(name="hub")
                surface2 = Surface(name="shroud")
                surface3 = Surface(name="stationary")
                _ = RotationVolume(
                    entities=[cylinder],
                    spacing_axial=20,
                    spacing_radial=0.2,
                    spacing_circumferential=20,
                    enclosed_entities=[surface1, surface2],
                    stationary_enclosed_entities=[
                        surface1,
                        surface3,
                    ],  # surface3 not in enclosed_entities
                )


def test_stationary_enclosed_entities_valid_subset():
    """Test that stationary_enclosed_entities works correctly when it's a valid subset."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cylinder",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            surface1 = Surface(name="hub")
            surface2 = Surface(name="shroud")
            # Should not raise when stationary_enclosed_entities is a valid subset
            _ = RotationVolume(
                entities=[cylinder],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[surface1, surface2],
                stationary_enclosed_entities=[surface1],  # Valid subset
            )

            # Should also work with all entities being stationary
            _ = RotationVolume(
                entities=[cylinder],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[surface1, surface2],
                stationary_enclosed_entities=[
                    surface1,
                    surface2,
                ],  # All entities stationary
            )

            # Should work with empty stationary_enclosed_entities (None)
            _ = RotationVolume(
                entities=[cylinder],
                spacing_axial=20,
                spacing_radial=0.2,
                spacing_circumferential=20,
                enclosed_entities=[surface1, surface2],
                stationary_enclosed_entities=None,
            )


def test_snappy_quality_metrics_validation():
    message = "Value must be less than or equal to 180 degrees."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.QualityMetrics(max_non_ortho=190 * u.deg)

    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        snappy.QualityMetrics(max_concave=190 * u.deg)

    snappy.QualityMetrics(max_non_ortho=90 * u.deg, max_concave=90 * u.deg)

    with SI_unit_system, pytest.raises(pd.ValidationError):
        snappy.QualityMetrics(max_boundary_skewness=-2 * u.deg)

    with SI_unit_system, pytest.raises(pd.ValidationError):
        snappy.QualityMetrics(max_internal_skewness=-2 * u.deg)

    snappy.QualityMetrics(
        max_boundary_skewness=23 * u.deg,
        max_internal_skewness=89 * u.deg,
        zmetric_threshold=0.9,
        feature_edge_deduplication_tolerance=0.1,
    )
    with pytest.raises(pd.ValidationError):
        snappy.QualityMetrics(zmetric_threshold=-0.1)
    with pytest.raises(pd.ValidationError):
        snappy.QualityMetrics(feature_edge_deduplication_tolerance=-0.1)

    snappy.QualityMetrics(zmetric_threshold=False, feature_edge_deduplication_tolerance=False)


def test_modular_workflow_zones_validation():
    message = "At least one zone defining the farfield is required."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[],
        )

    message = "When using `CustomZones` the `UserDefinedFarfield` will be ignored."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[
                UserDefinedFarfield(),
                CustomZones(
                    name="custom_zones",
                    entities=[SeedpointVolume(name="fluid", point_in_mesh=(0, 0, 0) * u.mm)],
                ),
            ],
        )

    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[
                CustomZones(
                    name="custom_zones",
                    entities=[
                        CustomVolume(
                            name="zone1",
                            boundaries=[Surface(name="face1"), Surface(name="face2")],
                        )
                    ],
                ),
                UserDefinedFarfield(),
            ],
        )

    message = "Only one `AutomatedFarfield` zone is allowed in `zones`."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[AutomatedFarfield(), AutomatedFarfield()],
        )

    message = "Only one `UserDefinedFarfield` zone is allowed in `zones`."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[UserDefinedFarfield(), UserDefinedFarfield()],
        )

    message = "Cannot use `AutomatedFarfield` and `UserDefinedFarfield` simultaneously."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[AutomatedFarfield(), UserDefinedFarfield()],
        )

    message = "`CustomZones` cannot be used with `AutomatedFarfield`."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[
                AutomatedFarfield(),
                CustomZones(
                    name="custom_zones",
                    entities=[SeedpointVolume(name="fluid", point_in_mesh=(0, 0, 0) * u.mm)],
                ),
            ],
        )

    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        ModularMeshingWorkflow(
            surface_meshing=snappy.SurfaceMeshingParams(
                defaults=snappy.SurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=2 * u.mm, gap_resolution=1 * u.mm
                )
            ),
            volume_meshing=VolumeMeshingParams(
                defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm)
            ),
            zones=[
                AutomatedFarfield(),
                CustomZones(
                    name="custom_zones",
                    entities=[
                        CustomVolume(
                            name="zone1",
                            boundaries=[Surface(name="face1"), Surface(name="face2")],
                        )
                    ],
                ),
            ],
        )


def test_uniform_project_only_with_snappy():
    refinement = UniformRefinement(
        entities=[Box(center=(0, 0, 0) * u.m, size=(1, 1, 1) * u.m, name="box")],
        spacing=0.1 * u.m,
        project_to_surface=True,
    )
    with SI_unit_system:
        params_snappy = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1 * u.mm,
                        max_spacing=2 * u.mm,
                        gap_resolution=1 * u.mm,
                    )
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm),
                    refinements=[refinement],
                ),
                zones=[
                    AutomatedFarfield(),
                ],
            )
        )

    params_snappy, errors, _ = validate_model(
        params_as_dict=params_snappy.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is None

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    AutomatedFarfield(),
                ],
                refinements=[refinement],
                defaults=MeshingDefaults(
                    curvature_resolution_angle=12 * u.deg,
                    boundary_layer_growth_rate=1.1,
                    boundary_layer_first_layer_thickness=1e-5 * u.m,
                ),
            )
        )

    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="VolumeMesh",
    )

    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, project_to_surface is supported only for snappyHexMesh."
    )
    assert errors[0]["loc"] == ("meshing", "refinements", 0, "UniformRefinement")


def test_resolve_face_boundary_only_in_gai_mesher():
    # raise when GAI is off
    with pytest.raises(
        pd.ValidationError,
        match=r"resolve_face_boundaries is only supported when geometry AI is used",
    ):
        with ValidationContext(SURFACE_MESH, non_gai_context):
            with CGS_unit_system:
                MeshingParams(
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=0.1,
                        resolve_face_boundaries=True,
                    )
                )


def test_surface_refinement_in_gai_mesher():
    # raise when both GAI and beta mesher are off
    with pytest.raises(
        pd.ValidationError,
        match=r"curvature_resolution_angle is only supported by the beta mesher or when geometry AI is enabled",
    ):
        with ValidationContext(SURFACE_MESH, non_gai_context):
            with CGS_unit_system:
                SurfaceRefinement(max_edge_length=0.1, curvature_resolution_angle=10 * u.deg)

    # raise when GAI is off
    with pytest.raises(
        pd.ValidationError,
        match=r"resolve_face_boundaries is only supported when geometry AI is used",
    ):
        with ValidationContext(SURFACE_MESH, non_gai_context):
            with CGS_unit_system:
                SurfaceRefinement(resolve_face_boundaries=True)

    # raise when no options are specified
    with pytest.raises(
        pd.ValidationError,
        match=r"SurfaceRefinement requires at least one of 'max_edge_length', 'curvature_resolution_angle', or 'resolve_face_boundaries' to be specified",
    ):
        with ValidationContext(SURFACE_MESH, non_gai_context):
            with CGS_unit_system:
                SurfaceRefinement(entities=Surface(name="testFace"))


def test_curvature_resolution_angle_requires_geometry_ai_or_beta_mesher():
    """Test that curvature_resolution_angle is supported when either geometry AI or beta mesher is enabled."""
    # Test 1: When both GAI and beta mesher are off, should raise
    with pytest.raises(
        pd.ValidationError,
        match=r"curvature_resolution_angle is only supported by the beta mesher or when geometry AI is enabled",
    ):
        with ValidationContext(SURFACE_MESH, non_gai_context):
            with CGS_unit_system:
                SurfaceRefinement(
                    entities=Surface(name="testFace"),
                    curvature_resolution_angle=15 * u.deg,
                )

    # Test 2: When curvature_resolution_angle is None, should not raise even if both are off
    with ValidationContext(SURFACE_MESH, non_gai_context):
        with CGS_unit_system:
            surface_ref = SurfaceRefinement(
                entities=Surface(name="testFace"),
                max_edge_length=0.1,
                curvature_resolution_angle=None,
            )
            assert surface_ref.curvature_resolution_angle is None

    # Test 3: When GAI is enabled, should work
    gai_context = ParamsValidationInfo({}, [])
    gai_context.use_geometry_AI = True

    with ValidationContext(SURFACE_MESH, gai_context):
        with CGS_unit_system:
            surface_ref = SurfaceRefinement(
                entities=Surface(name="testFace"),
                curvature_resolution_angle=20 * u.deg,
            )
            assert surface_ref.curvature_resolution_angle == 20 * u.deg

    # Test 4: When beta mesher is enabled, should work
    with ValidationContext(SURFACE_MESH, beta_mesher_context):
        with CGS_unit_system:
            surface_ref = SurfaceRefinement(
                entities=Surface(name="testFace"),
                curvature_resolution_angle=25 * u.deg,
            )
            assert surface_ref.curvature_resolution_angle == 25 * u.deg


def test_wind_tunnel_invalid_dimensions():
    with CGS_unit_system:
        # invalid floors
        with pytest.raises(
            pd.ValidationError,
            match=r"is not strictly increasing",
        ):
            # invalid range
            _ = StaticFloor(friction_patch_x_range=(-100, -200), friction_patch_width=42)

        with pytest.raises(
            pd.ValidationError,
            match=r"cannot have negative value",
        ):
            # invalid positive range
            _ = WheelBelts(
                central_belt_x_range=(-200, 256),
                central_belt_width=67,
                front_wheel_belt_x_range=(-30, 50),
                front_wheel_belt_y_range=(70, 120),
                rear_wheel_belt_x_range=(260, 380),
                rear_wheel_belt_y_range=(-5, 101),  # here
            )

        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than rear wheel belt minimum x",
        ):
            # front, rear belt x ranges overlap
            _ = WheelBelts(
                central_belt_x_range=(-200, 256),
                central_belt_width=67,
                front_wheel_belt_x_range=(-30, 263),  # here
                front_wheel_belt_y_range=(70, 120),
                rear_wheel_belt_x_range=(260, 380),
                rear_wheel_belt_y_range=(70, 120),
            )

        # invalid tunnels
        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than outlet x position",
        ):
            # inlet behind outlet
            _ = WindTunnelFarfield(
                inlet_x_position=200,
                outlet_x_position=182,
                floor_type=FullyMovingFloor(),
            )

        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than wind tunnel width",
        ):
            # friction patch too wide
            _ = WindTunnelFarfield(width=2025, floor_type=StaticFloor(friction_patch_width=9001))

        with pytest.raises(
            pd.ValidationError,
            match=r"must be greater than inlet x",
        ):
            # friction patch x min too small
            _ = WindTunnelFarfield(
                inlet_x_position=-2025,
                floor_type=StaticFloor(friction_patch_x_range=(-9001, 333)),
            )

        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than half of wind tunnel width",
        ):
            # wheel belt y outer too large
            _ = WindTunnelFarfield(
                width=538,
                floor_type=WheelBelts(
                    central_belt_x_range=(-200, 256),
                    central_belt_width=120,
                    front_wheel_belt_x_range=(-30, 50),
                    front_wheel_belt_y_range=(70, 270),
                    rear_wheel_belt_x_range=(260, 380),
                    rear_wheel_belt_y_range=(70, 120),
                ),
            )

        # legal, despite wheel belts being ahead/behind rather than left/right of central belt
        _ = WindTunnelFarfield(
            width=1024,
            floor_type=WheelBelts(
                central_belt_x_range=(-100, 105),
                central_belt_width=90.1,
                front_wheel_belt_x_range=(-30, 50),
                front_wheel_belt_y_range=(70, 123),
                rear_wheel_belt_x_range=(260, 380),
                rear_wheel_belt_y_range=(70, 120),
            ),
        )


def test_central_belt_width_validation():
    with CGS_unit_system:
        # Test central belt width larger than 2x front wheel belt inner edge
        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than or equal to twice the front wheel belt inner edge",
        ):
            _ = WheelBelts(
                central_belt_x_range=(-200, 256),
                central_belt_width=150,  # Width is 150
                front_wheel_belt_x_range=(-30, 50),
                front_wheel_belt_y_range=(
                    70,
                    120,
                ),  # Inner edge is 70, 2×70 = 140 < 150
                rear_wheel_belt_x_range=(260, 380),
                rear_wheel_belt_y_range=(80, 170),  # Inner edge is 80, 2×80 = 160 > 150
            )

        # Test central belt width larger than 2x rear wheel belt inner edge
        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than or equal to twice the rear wheel belt inner edge",
        ):
            _ = WheelBelts(
                central_belt_x_range=(-200, 256),
                central_belt_width=150,  # Width is 150
                front_wheel_belt_x_range=(-30, 50),
                front_wheel_belt_y_range=(
                    80,
                    170,
                ),  # Inner edge is 80, 2×80 = 160 > 150
                rear_wheel_belt_x_range=(260, 380),
                rear_wheel_belt_y_range=(70, 200),  # Inner edge is 70, 2×70 = 140 < 150
            )

        # Test central belt width larger than both inner edges
        with pytest.raises(
            pd.ValidationError,
            match=r"must be less than or equal to twice the front wheel belt inner edge",
        ):
            _ = WheelBelts(
                central_belt_x_range=(-200, 256),
                central_belt_width=200,  # Width is 200
                front_wheel_belt_x_range=(-30, 50),
                front_wheel_belt_y_range=(
                    90,
                    120,
                ),  # Inner edge is 90, 2×90 = 180 < 200
                rear_wheel_belt_x_range=(260, 380),
                rear_wheel_belt_y_range=(95, 140),  # Inner edge is 95, 2×95 = 190 < 200
            )

        # Legal: central belt width equal to 2x inner edges
        _ = WheelBelts(
            central_belt_x_range=(-200, 256),
            central_belt_width=140,  # Width is 140
            front_wheel_belt_x_range=(-30, 50),
            front_wheel_belt_y_range=(70, 170),  # Inner edge is 70, 2×70 = 140
            rear_wheel_belt_x_range=(260, 380),
            rear_wheel_belt_y_range=(70, 170),  # Inner edge is 70, 2×70 = 140
        )

        # Legal: central belt width less than 2x inner edges
        _ = WheelBelts(
            central_belt_x_range=(-200, 256),
            central_belt_width=100,  # Width is 100
            front_wheel_belt_x_range=(-30, 50),
            front_wheel_belt_y_range=(70, 170),  # Inner edge is 70, 2×70 = 140 > 100
            rear_wheel_belt_x_range=(260, 380),
            rear_wheel_belt_y_range=(80, 190),  # Inner edge is 80, 2×80 = 160 > 100
        )


def test_wind_tunnel_farfield_requires_geometry_ai():
    """Test that WindTunnelFarfield is only supported when Geometry AI is enabled."""
    # Test: When GAI is disabled, should raise error
    with pytest.raises(
        pd.ValidationError,
        match=r"WindTunnelFarfield is only supported when Geometry AI is enabled.",
    ):
        with ValidationContext(VOLUME_MESH, non_gai_context):
            with CGS_unit_system:
                WindTunnelFarfield()

    # Test: When GAI is enabled, should work
    gai_context = ParamsValidationInfo({}, [])
    gai_context.use_geometry_AI = True

    with ValidationContext(VOLUME_MESH, gai_context):
        with CGS_unit_system:
            farfield = WindTunnelFarfield()
            assert farfield.type == "WindTunnelFarfield"


def test_min_passage_size_requires_remove_hidden_geometry():
    """Test that min_passage_size can only be specified when remove_hidden_geometry is True."""
    gai_context = ParamsValidationInfo({}, [])
    gai_context.use_geometry_AI = True

    # Test 1: min_passage_size with remove_hidden_geometry=False should raise
    with pytest.raises(
        pd.ValidationError,
        match=r"'min_passage_size' can only be specified when 'remove_hidden_geometry' is True",
    ):
        with ValidationContext(SURFACE_MESH, gai_context):
            with SI_unit_system:
                MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=False,
                    min_passage_size=0.005 * u.m,
                )

    # Test 2: min_passage_size with remove_hidden_geometry=True should work
    with ValidationContext(SURFACE_MESH, gai_context):
        with SI_unit_system:
            defaults = MeshingDefaults(
                geometry_accuracy=0.01 * u.m,
                surface_max_edge_length=0.1 * u.m,
                remove_hidden_geometry=True,
                min_passage_size=0.005 * u.m,
            )
            assert defaults.min_passage_size == 0.005 * u.m
            assert defaults.remove_hidden_geometry is True

    # Test 3: remove_hidden_geometry=True without min_passage_size should work (it's optional)
    with ValidationContext(SURFACE_MESH, gai_context):
        with SI_unit_system:
            defaults = MeshingDefaults(
                geometry_accuracy=0.01 * u.m,
                surface_max_edge_length=0.1 * u.m,
                remove_hidden_geometry=True,
                min_passage_size=None,
            )
            assert defaults.min_passage_size is None
            assert defaults.remove_hidden_geometry is True


def test_meshing_defaults_octree_spacing_explicit():
    """Test that octree_spacing can be explicitly set on MeshingDefaults."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with SI_unit_system:
            defaults = MeshingDefaults(
                boundary_layer_first_layer_thickness=1e-5 * u.m,
                octree_spacing=2 * u.m,
            )
            assert defaults.octree_spacing is not None
            assert isinstance(defaults.octree_spacing, OctreeSpacing)
            assert defaults.octree_spacing.base_spacing == 2 * u.m
            # Verify indexing works through the field
            assert defaults.octree_spacing[0] == 2 * u.m
            assert defaults.octree_spacing[1] == 1 * u.m


def test_meshing_defaults_octree_spacing_auto_set_from_project_length_unit():
    """Test that octree_spacing is automatically set to 1 * project_length_unit when not specified."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            defaults = MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001,
            )
            # beta_mesher_context has project_length_unit = "mm"
            assert defaults.octree_spacing is not None
            assert isinstance(defaults.octree_spacing, OctreeSpacing)
            assert defaults.octree_spacing.base_spacing == 1 * u.mm


def test_meshing_defaults_octree_spacing_none_without_context():
    """Test that octree_spacing stays None when no validation context is active."""
    with CGS_unit_system:
        defaults = MeshingDefaults()
        assert defaults.octree_spacing is None


def test_meshing_defaults_octree_spacing_warning_no_project_length_unit():
    """Test that a validation warning is emitted when project_length_unit is None."""
    no_unit_context = ParamsValidationInfo({}, [])
    no_unit_context.is_beta_mesher = True
    no_unit_context.project_length_unit = None

    with ValidationContext(VOLUME_MESH, no_unit_context) as ctx:
        with CGS_unit_system:
            defaults = MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001,
            )
            assert defaults.octree_spacing is None

    warning_msgs = [w["msg"] if isinstance(w, dict) else str(w) for w in ctx.validation_warnings]
    assert any(
        "octree_spacing" in msg and "will not be set automatically" in msg for msg in warning_msgs
    )


def test_meshing_defaults_octree_spacing_negative_raises():
    """Test that negative octree_spacing raises a validation error."""
    with pytest.raises(pd.ValidationError):
        with SI_unit_system:
            MeshingDefaults(octree_spacing=-1 * u.m)


def test_meshing_defaults_octree_spacing_explicit_object():
    """Test that octree_spacing can be explicitly set as an OctreeSpacing object."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with SI_unit_system:
            spacing = OctreeSpacing(base_spacing=5 * u.m)
            defaults = MeshingDefaults(
                boundary_layer_first_layer_thickness=1e-5 * u.m,
                octree_spacing=spacing,
            )
            assert defaults.octree_spacing.base_spacing == 5 * u.m


def test_meshing_params_octree_check_skipped_for_non_beta():
    """Test that octree series check is skipped for non-beta mesher."""
    # Should not warn or raise — the validator returns early for non-beta
    with ValidationContext(VOLUME_MESH, non_beta_mesher_context) as ctx:
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cyl",
                outer_radius=10,
                height=20,
                axis=(0, 0, 1),
                center=(0, 0, 0),
            )
            MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                ),
                refinements=[
                    UniformRefinement(entities=[cylinder], spacing=0.3),
                ],
                volume_zones=[AutomatedFarfield()],
            )
    # No octree-related warnings for non-beta mesher
    warning_msgs = [w["msg"] if isinstance(w, dict) else str(w) for w in ctx.validation_warnings]
    assert not any("octree series" in msg for msg in warning_msgs)


def test_meshing_params_octree_check_warns_for_non_aligned_spacing(capsys):
    """Test that octree series check warns when spacing doesn't align with octree series."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cyl",
                outer_radius=10,
                height=20,
                axis=(0, 0, 1),
                center=(0, 0, 0),
            )
            MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    octree_spacing=1 * u.mm,
                ),
                refinements=[
                    # 0.3 mm is not a power-of-2 fraction of 1 mm
                    UniformRefinement(entities=[cylinder], spacing=0.3 * u.mm),
                ],
                volume_zones=[AutomatedFarfield()],
            )
    captured = capsys.readouterr()
    captured_text = " ".join(captured.out.split())
    assert "will be cast to the first lower refinement" in captured_text


def test_meshing_params_octree_check_no_warn_for_aligned_spacing():
    """Test that octree series check does not warn for aligned spacing."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cyl",
                outer_radius=10,
                height=20,
                axis=(0, 0, 1),
                center=(0, 0, 0),
            )
            # 0.5 mm = 1mm * 2^-1, so this is aligned
            MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    octree_spacing=1 * u.mm,
                ),
                refinements=[
                    UniformRefinement(entities=[cylinder], spacing=0.5 * u.mm),
                ],
                volume_zones=[AutomatedFarfield()],
            )


def test_meshing_params_octree_check_skipped_when_octree_spacing_none():
    """Test that octree check is skipped when octree_spacing is None."""
    no_unit_context = ParamsValidationInfo({}, [])
    no_unit_context.is_beta_mesher = True
    no_unit_context.project_length_unit = None

    with ValidationContext(VOLUME_MESH, no_unit_context):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="cyl",
                outer_radius=10,
                height=20,
                axis=(0, 0, 1),
                center=(0, 0, 0),
            )
            # Should not raise — validator just logs a warning and skips
            MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                ),
                refinements=[
                    UniformRefinement(entities=[cylinder], spacing=0.3),
                ],
                volume_zones=[AutomatedFarfield()],
            )


def test_meshing_params_octree_check_multiple_refinements():
    """Test that octree series check runs on all UniformRefinements."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            cylinder1 = Cylinder(
                name="cyl1",
                outer_radius=10,
                height=20,
                axis=(0, 0, 1),
                center=(0, 0, 0),
            )
            cylinder2 = Cylinder(
                name="cyl2",
                outer_radius=5,
                height=10,
                axis=(0, 0, 1),
                center=(1, 0, 0),
            )
            # Both spacings are powers of 2 of the base, should not warn
            MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    octree_spacing=1 * u.mm,
                ),
                refinements=[
                    UniformRefinement(entities=[cylinder1], spacing=0.25 * u.mm),
                    UniformRefinement(entities=[cylinder2], spacing=0.125 * u.mm),
                ],
                volume_zones=[AutomatedFarfield()],
            )


def test_meshing_params_octree_check_no_refinements():
    """Test that octree check does not fail when there are no refinements."""
    with ValidationContext(VOLUME_MESH, beta_mesher_context):
        with CGS_unit_system:
            MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    octree_spacing=1 * u.mm,
                ),
                refinements=[],
                volume_zones=[AutomatedFarfield()],
            )


def test_per_face_min_passage_size_warning_without_remove_hidden_geometry():
    """Test that per-face min_passage_size on GeometryRefinement warns when remove_hidden_geometry is disabled."""

    # Test 1: min_passage_size on GeometryRefinement with remove_hidden_geometry=False → warning
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=False,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=0.01 * u.m,
                        min_passage_size=0.05 * u.m,
                        faces=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True, use_inhouse_mesher=True, project_length_unit=1 * u.m
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert len(warnings) == 1
    assert "min_passage_size" in warnings[0]["msg"]
    assert "remove_hidden_geometry" in warnings[0]["msg"]

    # Test 2: min_passage_size on GeometryRefinement with remove_hidden_geometry=True → no warning
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=0.01 * u.m,
                        min_passage_size=0.05 * u.m,
                        faces=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True, use_inhouse_mesher=True, project_length_unit=1 * u.m
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []

    # Test 3: GeometryRefinement without min_passage_size, remove_hidden_geometry=False → no warning
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=False,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=0.01 * u.m,
                        faces=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True, use_inhouse_mesher=True, project_length_unit=1 * u.m
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []
