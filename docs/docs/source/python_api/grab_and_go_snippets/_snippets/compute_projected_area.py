import flow360 as fl

project = fl.Project.from_cloud("PROJECT_ID_HERE")
geometry = project.geometry

with fl.create_draft(
    new_run_from=geometry,
    face_grouping="face_grouping_tag",
) as draft:
    wing_surfaces = draft.surfaces["wing_*"]

    # Compute a concrete preview immediately.
    preview_area = fl.measure.projected_area(
        draft,
        surfaces=wing_surfaces,
        direction="Z",
    )
    print("projected area:", preview_area)

    # Store an automatic recipe in the simulation parameters. Submission from
    # this active draft recomputes the value before validation and upload.
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                area=fl.ProjectedArea(
                    surfaces=wing_surfaces,
                    direction="Z",
                )
            )
        )

    # Configure the remaining simulation parameters and submit within this
    # draft context so the selected geometry and tessellation remain available.
