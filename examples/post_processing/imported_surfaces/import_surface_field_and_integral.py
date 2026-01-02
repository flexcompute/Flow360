import flow360 as fl
from flow360.examples import ObliqueChannel

ObliqueChannel.get_files()

project = fl.Project.from_volume_mesh(ObliqueChannel.mesh_filename, name="Cartesian channel mesh")

volume_mesh = project.volume_mesh

normal_imported_surface = project.import_surface_mesh(
    ObliqueChannel.extra["rectangle_normal"], name="normal"
)
oblique_imported_surface = project.import_surface_mesh(
    ObliqueChannel.extra["rectangle_oblique"], name="oblique"
)
imported_surfaces = [normal_imported_surface, oblique_imported_surface]

with fl.create_draft(
    new_run_from=volume_mesh,
    imported_surfaces=imported_surfaces,
) as draft:
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
            surfaces=draft.surfaces["VOLUME/LEFT"],
        )
        params = fl.SimulationParams(
            operating_condition=op,
            models=[
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
                    turbulence_model_solver=fl.NoneSolver(),
                ),
                fl.Inflow(
                    entities=[draft.surfaces["VOLUME/LEFT"]],
                    total_temperature=op.thermal_state.temperature * 1.018,
                    velocity_direction=(1.0, 0.0, 0.0),
                    spec=fl.MassFlowRate(
                        value=op.velocity_magnitude * op.thermal_state.density * (0.2 * fl.u.m**2)
                    ),
                ),
                fl.Outflow(
                    entities=[draft.surfaces["VOLUME/RIGHT"]],
                    spec=fl.Pressure(op.thermal_state.pressure),
                ),
                fl.SlipWall(
                    entities=[
                        draft.surfaces["VOLUME/FRONT"],
                        draft.surfaces["VOLUME/BACK"],
                        draft.surfaces["VOLUME/TOP"],
                        draft.surfaces["VOLUME/BOTTOM"],
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
                        draft.surfaces["VOLUME/FRONT"],
                        draft.imported_surfaces["normal"],
                        draft.imported_surfaces["oblique"],
                    ],
                ),
                fl.SurfaceIntegralOutput(
                    name="MassFlowRateImportedSurface",
                    output_fields=[massFlowRate],
                    surfaces=[
                        draft.imported_surfaces["normal"],
                        draft.imported_surfaces["oblique"],
                    ],
                ),
            ],
        )
    project.run_case(params, "test_imported_surfaces_field_and_integral")
