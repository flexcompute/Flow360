import flow360 as fl
from flow360.examples import ObliqueChannel

ObliqueChannel.get_files()

project = fl.Project.from_volume_mesh(
    ObliqueChannel.mesh_filename,
    name="Cartesian channel mesh",
)

volume_mesh = project.volume_mesh

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
    massFlowRateIntegral = fl.SurfaceIntegralOutput(
        name="MassFluxIntegral",
        output_fields=[massFlowRate],
        surfaces=volume_mesh["VOLUME/LEFT"],
    )
    params = fl.SimulationParams(
        operating_condition=op,
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
                turbulence_model_solver=fl.NoneSolver(),
            ),
            fl.Inflow(
                entities=[volume_mesh["VOLUME/LEFT"]],
                total_temperature=op.thermal_state.temperature * 1.018,
                velocity_direction=(1.0, 0.0, 0.0),
                spec=fl.MassFlowRate(
                    value=op.velocity_magnitude * op.thermal_state.density * (0.2 * fl.u.m**2)
                ),
            ),
            fl.Outflow(
                entities=[volume_mesh["VOLUME/RIGHT"]],
                spec=fl.Pressure(op.thermal_state.pressure),
            ),
            fl.SlipWall(
                entities=[
                    volume_mesh["VOLUME/FRONT"],
                    volume_mesh["VOLUME/BACK"],
                    volume_mesh["VOLUME/TOP"],
                    volume_mesh["VOLUME/BOTTOM"],
                ]
            ),
        ],
        time_stepping=fl.Steady(),
        outputs=[
            fl.VolumeOutput(
                output_format="paraview",
                output_fields=["primitiveVars"],
            ),
            fl.SurfaceOutput(
                output_fields=[
                    fl.solution.velocity,
                    fl.solution.Cp,
                ],
                surfaces=[
                    fl.ImportedSurface(
                        name="normal", file_name=ObliqueChannel.extra["rectangle_normal"]
                    ),
                    fl.ImportedSurface(
                        name="oblique", file_name=ObliqueChannel.extra["rectangle_oblique"]
                    ),
                ],
            ),
            fl.SurfaceIntegralOutput(
                name="MassFlowRateImportedSurface",
                output_fields=[massFlowRate],
                surfaces=[
                    fl.ImportedSurface(
                        name="normal", file_name=ObliqueChannel.extra["rectangle_normal"]
                    ),
                    fl.ImportedSurface(
                        name="oblique", file_name=ObliqueChannel.extra["rectangle_oblique"]
                    ),
                ],
            ),
        ],
    )
project.run_case(params, "test_imported_surfaces_field_and_integral")
