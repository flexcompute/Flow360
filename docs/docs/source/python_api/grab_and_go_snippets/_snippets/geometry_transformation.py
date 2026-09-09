import flow360 as fl

project = fl.Project.from_cloud("PROJECT_ID_HERE")
geometry = project.geometry  # or project.volume_mesh, project.surface_mesh

with fl.create_draft(
    new_run_from=geometry,
    face_grouping="face_grouping_tag",
    edge_grouping="edge_grouping_tag",
) as draft:

    draft.coordinate_systems.assign(
        entities=draft.body_groups["body_group_1"],
        coordinate_system=fl.CoordinateSystem(
            name="body_group_1_transform",
            origin=[0, 0, 0] * fl.u.mm,
            axis_of_rotation=(0, 0, 1),
            angle_of_rotation=0 * fl.u.deg,
            scale=(1.0, 1.0, 1.0),
            translation=[20, 0, 0] * fl.u.mm
        )
    )

    params = fl.SimulationParams(...)

    project.run_case(params=params, name="Baseline case")

    # Modify the rotation angle for the next case
    draft.coordinate_systems.get_by_name("body_group_1_transform").angle_of_rotation = 5 * fl.u.deg

    project.run_case(params=params, name="Rotated 5 deg case")
