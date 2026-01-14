import json
import os

from flow360.component.geometry import GeometryMeta
from flow360.component.project_utils import (
    set_up_params_for_uploading,
    validate_params_with_context,
)
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.framework.entity_selector import (
    collect_and_tokenize_selectors_in_place,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.primitives import MirroredSurface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.test_solver_translator import translate_and_compare


def test_mirrored_surface_translation():
    import flow360 as fl

    metadata_path = os.path.join(os.path.dirname(__file__), "data", "gai_mirror_entity_info")
    geometry = fl.Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",  # placeholder UUID
        local_storage_path=metadata_path,
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="geometry_name_placeholder",
                cloud_path_prefix="s3_path_placeholder",
                status="processed",
            )
        ),
    )

    geometry.internal_registry = geometry._entity_info.get_persistent_entity_registry(
        geometry.internal_registry
    )

    mesh_unit = 1 * fl.u.m
    half_body_factor = 0.5
    wheel_radius = 0.35
    wheel_width = 0.25
    wheel_ref_area = 2 * wheel_radius * wheel_width * half_body_factor

    with fl.create_draft(new_run_from=geometry, face_grouping="faceId") as draft:
        body_groups = list(draft.body_groups)
        body_group = body_groups[0]
        draft.coordinate_systems.assign(
            entities=body_group,
            coordinate_system=fl.CoordinateSystem(
                name="shift",
                reference_point=[0, 0, 0] * mesh_unit,
                axis_of_rotation=(0, 0, 1),
                angle_of_rotation=0 * fl.u.deg,
                scale=(1.0, 1.0, 1.0),
                translation=[0.067, -0.933, 0] * mesh_unit,
            ),
        )
        # Use fixed IDs for test reproducibility
        mirror_plane_id = "mirror-plane-test-001"
        plane = fl.MirrorPlane(
            name="mirror_local",
            normal=(1, 0, 0),
            center=(1.6, 0, 0) * fl.u.m,
            private_attribute_id=mirror_plane_id,
        )
        _, draft_mirrored_surfaces = draft.mirror.create_mirror_of(
            entities=body_group, mirror_plane=plane
        )

        # Use the mirrored surfaces owned by the draft's mirror_status/registry.
        # This avoids introducing foreign mirrored entities that are not tracked by the draft.
        expected_mirrored_names = {"Curved_<mirror>", "TopCap_<mirror>", "BottomCap_<mirror>"}
        mirrored_surfaces = [
            mirrored
            for mirrored in draft_mirrored_surfaces
            if mirrored.name in expected_mirrored_names
        ]
        assert {m.name for m in mirrored_surfaces} == expected_mirrored_names

        wind_tunnel = fl.WindTunnelFarfield(
            width=10 * mesh_unit,
            height=10 * mesh_unit,
            inlet_x_position=-5 * mesh_unit,
            outlet_x_position=15 * mesh_unit,
            floor_z_position=0 * mesh_unit,
            floor_type=fl.WheelBelts(
                central_belt_x_range=(-1, 6) * mesh_unit,
                central_belt_width=1.2 * mesh_unit,
                front_wheel_belt_x_range=(-0.3, 0.5) * mesh_unit,
                front_wheel_belt_y_range=(0.7, 1.2) * mesh_unit,
                rear_wheel_belt_x_range=(2.6, 3.8) * mesh_unit,
                rear_wheel_belt_y_range=(0.7, 1.2) * mesh_unit,
            ),
        )

        meshing_params = fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=0.2 * mesh_unit,
                surface_max_aspect_ratio=10,
                curvature_resolution_angle=15 * fl.u.deg,
                geometry_accuracy=1e-2 * mesh_unit,
                boundary_layer_first_layer_thickness=1e-4 * mesh_unit,
                boundary_layer_growth_rate=1.2,
                planar_face_tolerance=1e-6,
            ),
            volume_zones=[wind_tunnel],
        )
        with SI_unit_system:
            simulation_params = fl.SimulationParams(
                meshing=meshing_params,
                reference_geometry=fl.ReferenceGeometry(
                    area=wheel_ref_area * mesh_unit * mesh_unit,
                    moment_length=(1.0, 1.0, 1.0) * mesh_unit,
                    moment_center=(0, 0, wheel_radius) * mesh_unit,
                ),
                operating_condition=fl.AerospaceCondition(
                    velocity_magnitude=30 * mesh_unit / fl.u.s
                ),
                models=[
                    fl.Fluid(
                        navier_stokes_solver=fl.NavierStokesSolver(
                            linear_solver=fl.LinearSolver(max_iterations=50),
                            absolute_tolerance=1e-9,
                            low_mach_preconditioner=True,
                        ),
                        turbulence_model_solver=fl.SpalartAllmaras(absolute_tolerance=1e-8),
                    ),
                    fl.SlipWall(
                        entities=[
                            wind_tunnel.left,
                            wind_tunnel.right,
                            wind_tunnel.ceiling,
                        ]
                    ),
                    fl.Wall(
                        entities=[
                            geometry["Curved"],
                            geometry["TopCap"],
                            geometry["BottomCap"],
                            wind_tunnel.floor,
                        ]
                        + mirrored_surfaces,
                        use_wall_function=False,
                    ),
                    fl.Wall(
                        entities=[
                            wind_tunnel.central_belt,
                            wind_tunnel.front_wheel_belts,
                            wind_tunnel.rear_wheel_belts,
                        ],
                        velocity=[
                            30 * mesh_unit / fl.u.s,
                            0 * mesh_unit / fl.u.s,
                            0 * mesh_unit / fl.u.s,
                        ],
                        use_wall_function=True,
                    ),
                    fl.Freestream(
                        entities=[
                            wind_tunnel.inlet,
                            wind_tunnel.outlet,
                        ]
                    ),
                ],
                time_stepping=fl.Steady(CFL=fl.AdaptiveCFL(max=1000), max_steps=100),
                outputs=[
                    fl.VolumeOutput(
                        output_format="paraview",
                        output_fields=["primitiveVars", "Mach", "mutRatio"],
                    ),
                    fl.SurfaceOutput(
                        entities=[
                            geometry["Curved"],
                            geometry["TopCap"],
                            geometry["BottomCap"],
                            wind_tunnel.floor,
                            wind_tunnel.central_belt,
                            wind_tunnel.front_wheel_belts,
                            wind_tunnel.rear_wheel_belts,
                        ]
                        + mirrored_surfaces,
                        output_format="paraview",
                        output_fields=["primitiveVars", "Cp", "Cf", "yPlus"],
                    ),
                ],
            )

        # Then it goes through set_up_params_for_uploading
        processed_params = set_up_params_for_uploading(
            geometry, mesh_unit, simulation_params, True, True
        )

        params_dict = processed_params.model_dump(mode="json", exclude_none=True)
        params_dict = collect_and_tokenize_selectors_in_place(params_dict)

        simulation_param, _validation_errors, _ = validate_model(
            params_as_dict=params_dict,
            validated_by=ValidationCalledBy.PIPELINE,
            root_item_type="Geometry",
            validation_level="All",
        )

        # Dump and compare with surface mesh ref JSON
        surface_mesh_param, err, warnings = validate_params_with_context(
            simulation_param, "Geometry", "SurfaceMesh"
        )
        assert err is None, f"Surface mesh validation error: {err}"
        assert warnings == [], f"Unexpected warnings for surface mesh validation: {warnings}"
        surface_mesh_translated = get_surface_meshing_json(surface_mesh_param, mesh_unit=mesh_unit)
        surface_mesh_ref_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ref",
            "Flow360_mirrored_surface_meshing.json",
        )
        with open(surface_mesh_ref_file, "r") as fh:
            surface_mesh_ref_dict = json.load(fh)
        # Ignore private_attribute_id when comparing since they may be regenerated
        assert compare_values(
            surface_mesh_ref_dict, surface_mesh_translated, ignore_keys={"private_attribute_id"}
        ), "Surface mesh translation does not match reference"

        # Dump and compare with volume mesh ref JSON
        volume_mesh_param, err, warnings = validate_params_with_context(
            simulation_param, "Geometry", "VolumeMesh"
        )
        assert err is None, f"Volume mesh validation error: {err}"
        assert warnings == [], f"Unexpected warnings for volume mesh validation: {warnings}"
        volume_mesh_translated = get_volume_meshing_json(volume_mesh_param, mesh_unit=mesh_unit)
        volume_mesh_ref_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ref",
            "Flow360_mirrored_volume_meshing.json",
        )
        with open(volume_mesh_ref_file, "r") as fh:
            volume_mesh_ref_dict = json.load(fh)
        assert compare_values(
            volume_mesh_ref_dict, volume_mesh_translated
        ), "Volume mesh translation does not match reference"

        with open(os.path.join(metadata_path, "volume_mesh_meta_data.json"), "r") as f:
            volume_mesh_meta_data = json.load(f)

        simulation_param._update_param_with_actual_volume_mesh_meta(volume_mesh_meta_data)
        translate_and_compare(
            simulation_param,
            mesh_unit=mesh_unit,
            ref_json_file="Flow360_mirrored_surface_translation.json",
            debug=False,
        )
