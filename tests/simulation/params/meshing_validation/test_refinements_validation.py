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
from flow360.component.simulation.primitives import Box, Cylinder
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def _make_snappy_params_with_volume_uniform_refinement(refinement):
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

    assert errors is not None
    error_messages = [error["msg"] for error in errors]
    assert any("angle_of_rotation" in msg or "axes aligned" in msg for msg in error_messages)


def test_volume_uniform_refinement_hollow_cylinder_project_to_surface():
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

    assert errors is not None
    error_messages = [error["msg"] for error in errors]
    assert any("inner_radius" in msg or "full cylinders" in msg for msg in error_messages)


def test_volume_uniform_refinement_cylinder_none_inner_radius_project_to_surface():
    full_cylinder = Cylinder(
        name="full_cyl_none",
        inner_radius=None,
        outer_radius=7 * u.mm,
        axis=[0, 0, 1],
        center=[0, 0, 0] * u.m,
        height=10 * u.mm,
    )
    refinement = UniformRefinement(
        spacing=5 * u.mm,
        entities=[full_cylinder],
        project_to_surface=True,
    )
    params = _make_snappy_params_with_volume_uniform_refinement(refinement)

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is None


def test_volume_uniform_refinement_default_project_to_surface():
    rotated_box = Box(
        center=[0, 0, 0] * u.m,
        size=[1, 1, 1] * u.m,
        axis_of_rotation=[0, 0, 1],
        angle_of_rotation=90 * u.deg,
        name="rotated_box_default",
    )
    refinement = UniformRefinement(spacing=5 * u.mm, entities=[rotated_box])
    params = _make_snappy_params_with_volume_uniform_refinement(refinement)

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )

    assert errors is not None
    error_messages = [error["msg"] for error in errors]
    assert any("angle_of_rotation" in msg or "axes aligned" in msg for msg in error_messages)


def test_volume_uniform_refinement_project_to_surface_false_skips_validation():
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
