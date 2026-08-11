import flow360 as fl

# Load an existing project that already has a volume mesh
project = fl.Project.from_cloud("YOUR_PROJECT_ID")
volume_mesh = project.volume_mesh

# Import one or more surface mesh files (STL, CGNS, or UGRID)
surface_a = project.import_surface_mesh("surface_a.stl", name="surface_a")
surface_b = project.import_surface_mesh("surface_b.stl", name="surface_b")

with fl.create_draft(
    new_run_from=volume_mesh,
    imported_surfaces=[surface_a, surface_b],
) as draft:
    with fl.SI_unit_system:
        # Define a user variable to compute local mass flux at each surface node
        mass_flux = fl.UserVariable(
            name="MassFlux",
            value=fl.solution.density
            * fl.math.dot(fl.solution.velocity, fl.solution.node_unit_normal),
        )

        params = fl.SimulationParams(
            operating_condition=fl.AerospaceCondition(velocity_magnitude=10 * fl.u.m / fl.u.s),
            models=[...],
            time_stepping=fl.Steady(),
            outputs=[
                # Extract flow field quantities on the imported surfaces
                fl.SurfaceOutput(
                    output_fields=[fl.solution.velocity, fl.solution.Cp],
                    surfaces=[
                        draft.imported_surfaces["surface_a"],
                        draft.imported_surfaces["surface_b"],
                    ],
                ),
                # Integrate the mass flux over each imported surface
                fl.SurfaceIntegralOutput(
                    name="MassFluxIntegral",
                    output_fields=[mass_flux],
                    surfaces=[
                        draft.imported_surfaces["surface_a"],
                        draft.imported_surfaces["surface_b"],
                    ],
                ),
            ],
        )
    project.run_case(params, name="imported_surface_outputs")
