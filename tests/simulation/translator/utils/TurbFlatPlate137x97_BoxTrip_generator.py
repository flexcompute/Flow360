import pytest

import flow360.component.simulation.models.material as material
import flow360.component.simulation.models.solver_numerics as numerics
import flow360.component.simulation.models.surface_models as srf_models
import flow360.component.simulation.models.volume_models as vol_models
import flow360.component.simulation.operating_condition.operating_condition as op_condition
import flow360.component.simulation.outputs.outputs as outputs
import flow360.component.simulation.primitives as primitives
import flow360.component.simulation.simulation_params as params
import flow360.component.simulation.time_stepping.time_stepping as stepping
import flow360.component.simulation.units as units


@pytest.fixture()
def create_turb_flat_plate_box_trip_param():

    with units.SI_unit_system:

        thermal_state = op_condition.ThermalState(
            temperature=300 * units.K,
            density=1.225 * units.kg / units.m**3,
            material=material.Air(dynamic_viscosity=1.70138e-5 * units.Pa * units.s),
        )

        param = params.SimulationParams(
            operating_condition=op_condition.GenericReferenceCondition.from_mach(
                mach=0.2, thermal_state=thermal_state
            ),
            models=[
                vol_models.Fluid(
                    navier_stokes_solver=numerics.NavierStokesSolver(
                        linear_solver=numerics.LinearSolver(max_iterations=50),
                        absolute_tolerance=1e-10,
                        low_mach_preconditioner=True,
                    ),
                    turbulence_model_solver=numerics.SpalartAllmaras(
                        linear_solver=numerics.LinearSolver(max_iterations=50),
                        absolute_tolerance=1e-8,
                    ),
                    transition_model_solver=numerics.TransitionModelSolver(
                        linear_solver=numerics.LinearSolver(max_iterations=50),
                        absolute_tolerance=1e-8,
                        update_jacobian_frequency=1,
                        equation_evaluation_frequency=1,
                        turbulence_intensity_percent=0.04,
                        trip_regions=[
                            primitives.Box.from_principal_axes(
                                name="Box",
                                center=(0.25, -0.5, 0.025 + 0.0001),
                                size=(0.05, 1.0, 0.05),
                                axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                            )
                        ],
                    ),
                ),
                srf_models.SlipWall(
                    entities=[
                        primitives.Surface(name="1"),
                        primitives.Surface(name="2"),
                        primitives.Surface(name="7"),
                    ]
                ),
                srf_models.Inflow(
                    entities=[primitives.Surface(name="3")],
                    spec=srf_models.TotalPressure(value=1.02828 * thermal_state.pressure),
                    total_temperature=1.008 * thermal_state.temperature,
                ),
                srf_models.Outflow(
                    entities=[primitives.Surface(name="4")],
                    spec=srf_models.Pressure(thermal_state.pressure),
                ),
                srf_models.SymmetryPlane(entities=[primitives.Surface(name="5")]),
                srf_models.Wall(entities=[primitives.Surface(name="6")]),
            ],
            time_stepping=stepping.Steady(
                CFL=stepping.RampCFL(initial=1, final=100, ramp_steps=200), max_steps=20000
            ),
            outputs=[
                outputs.VolumeOutput(
                    output_format="paraview",
                    output_fields=[
                        "primitiveVars",
                        "residualNavierStokes",
                        "vorticity",
                        "solutionTurbulence",
                        "mutRatio",
                        "solutionTransition",
                    ],
                ),
                outputs.SurfaceOutput(
                    entities=[primitives.Surface(name="6")],
                    output_format="paraview",
                    output_fields=["Cf"],
                ),
            ],
            private_attribute_asset_cache=params.AssetCache(project_length_unit=1 * units.m),
        )

    return param
