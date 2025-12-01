import pytest

import flow360 as fl

# copied from analyticWindTunnelToCase


@pytest.fixture
def create_windtunnel_params():
    with fl.SI_unit_system:
        wind_tunnel = fl.WindTunnelFarfield(
            width=10 * fl.u.m,
            height=10 * fl.u.m,
            inlet_x_position=-5 * fl.u.m,
            outlet_x_position=15 * fl.u.m,
            floor_z_position=0 * fl.u.m,
            floor_type=fl.WheelBelts(
                central_belt_x_range=(-1, 6) * fl.u.m,
                central_belt_width=1.2 * fl.u.m,
                front_wheel_belt_x_range=(-0.3, 0.5) * fl.u.m,
                front_wheel_belt_y_range=(0.7, 1.2) * fl.u.m,
                rear_wheel_belt_x_range=(2.6, 3.8) * fl.u.m,
                rear_wheel_belt_y_range=(0.7, 1.2) * fl.u.m,
            ),
        )
        meshing_params = fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_aspect_ratio=10,
                curvature_resolution_angle=15 * fl.u.deg,
                geometry_accuracy=1e-2 * fl.u.m,
                boundary_layer_first_layer_thickness=1e-4 * fl.u.m,
                boundary_layer_growth_rate=1.2,
                planar_face_tolerance=1e-3,
            ),
            volume_zones=[wind_tunnel],
        )

        simulation_params = fl.SimulationParams(
            meshing=meshing_params,
            operating_condition=fl.AerospaceCondition(velocity_magnitude=30 * fl.u.m / fl.u.s),
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
                        wind_tunnel.left(),
                        wind_tunnel.right(),
                        wind_tunnel.ceiling(),
                    ]
                ),
                fl.Wall(entities=[wind_tunnel.floor()], use_wall_function=False),
                fl.Wall(
                    entities=[
                        wind_tunnel.central_belt(),
                        wind_tunnel.front_wheel_belts(),
                        wind_tunnel.rear_wheel_belts(),
                    ],
                    velocity=[30 * fl.u.m / fl.u.s, 0 * fl.u.m / fl.u.s, 0 * fl.u.m / fl.u.s],
                    use_wall_function=True,
                ),
                fl.Freestream(entities=[wind_tunnel.inlet(), wind_tunnel.outlet()]),
            ],
            time_stepping=fl.Steady(
                CFL=fl.AdaptiveCFL(max=1000),
                max_steps=100,  # reduced to speed up simulation; might not converge
            ),
            outputs=[
                fl.VolumeOutput(
                    output_format="paraview", output_fields=["primitiveVars", "Mach", "mutRatio"]
                ),
                fl.SurfaceOutput(
                    entities=[
                        wind_tunnel.floor(),
                        wind_tunnel.central_belt(),
                        wind_tunnel.front_wheel_belts(),
                        wind_tunnel.rear_wheel_belts(),
                    ],
                    output_format="paraview",
                    output_fields=["primitiveVars", "Cp", "Cf", "yPlus"],
                ),
            ],
        )
    return simulation_params
