"""
Test specifying velocity direction for Inflow boundary with MassFlowRate.
The Inflow has non-X normal component but the velocity direction is (1, 0, 0)
Therefore the specified velocity direction should contain X only component at inlet.
"""

import flow360 as fl

fl.Env.dev.active()

def createBaseParams(volumeMesh):
    with fl.SI_unit_system:
        op = fl.GenericReferenceCondition.from_mach(
            mach=0.3,
        )
        massFlowRate = fl.UserVariable(
            name="MassFluxProjected",
            value=-1
            * fl.solution.density
            * fl.math.dot(fl.solution.velocity, fl.solution.node_unit_normal),
        )
        params = fl.SimulationParams(
            operating_condition=op,
            models=[
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
                    turbulence_model_solver=fl.NoneSolver(),
                    stopping_criterion = [],
                ),
                fl.Inflow(
                    entities=[volumeMesh["VOLUME/LEFT"]],
                    total_temperature=op.thermal_state.temperature * 1.018,
                    velocity_direction=(1.0, 0.0, 0.0),
                    spec=fl.MassFlowRate(
                        value=op.velocity_magnitude
                        * op.thermal_state.density
                        * (0.2 * fl.u.m**2)
                    ),
                ),
                fl.Outflow(
                    entities=[volumeMesh["VOLUME/RIGHT"]],
                    spec=fl.Pressure(op.thermal_state.pressure),
                ),
                fl.SlipWall(
                    entities=[
                        volumeMesh["VOLUME/FRONT"],
                        volumeMesh["VOLUME/BACK"],
                        volumeMesh["VOLUME/TOP"],
                        volumeMesh["VOLUME/BOTTOM"],
                    ]
                ),
            ],
            time_stepping=fl.Steady(max_steps=1000),
            outputs=[
                fl.VolumeOutput(
                    output_format="paraview",
                    output_fields=["primitiveVars"],
                ),
                fl.ImportedSurfaceOutput(
                    output_fields=[
                        fl.solution.velocity,
                        fl.solution.Cp,
                    ],
                    surfaces=[
                        fl.ImportedSurface(
                            name="normal", file_name="rectangle_normal.cgns"
                        ),
                        fl.ImportedSurface(
                            name="oblique", file_name="rectangle_oblique.cgns"
                        ),
                    ],
                ),
                fl.ImportedSurfaceIntegralOutput(
                    name="MassFlowRateImportedSurface",
                    output_fields=[massFlowRate],
                    surfaces=[
                        fl.ImportedSurface(
                            name="normal", file_name="rectangle_normal.cgns"
                        ),
                        fl.ImportedSurface(
                            name="oblique", file_name="rectangle_oblique.cgns"
                        ),
                    ],
                ),
            ],
        )
        return params

if __name__ == "__main__":
    meshFile = "cartesian_2d_mesh.oblique.cgns"
    project = fl.Project.from_volume_mesh(meshFile, name="Test Imported Surface Output", solver_version="release-25.7.1")
    vm = project.volume_mesh
    params = createBaseParams(vm)
    case = project.run_case(params, "Run_imported_surface_output")


