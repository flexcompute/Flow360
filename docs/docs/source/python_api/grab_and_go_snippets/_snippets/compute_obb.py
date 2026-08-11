import flow360 as fl

project = fl.Project.from_cloud("PROJECT_ID_HERE")
geometry = project.geometry

with fl.create_draft(
    new_run_from=geometry,
    face_grouping="face_grouping_tag",
) as draft:

    # Select the surfaces that make up the wheel.
    wheel_surfaces = draft.surfaces["wheel_*"]

    # Fit the oriented bounding box to those surfaces. The result holds the
    # box geometry only: center, principal axes and half-extents.
    obb = draft.compute_obb(wheel_surfaces)

    # Derive the rotation axis and radius from the box. Pass a known
    # direction with rotation_axis_hint (or an explicit axis_index). When
    # neither is given the axis is inferred from the most circular
    # cross-section and a warning is emitted.
    rotation = obb.get_rotation_axis_and_radius(rotation_axis_hint=[0, 1, 0])

    print("center:", obb.center)
    print("rotation axis:", rotation.axis_of_rotation)
    print("averaged radius:", rotation.averaged_radius)

    # Printing the result also shows how the radius was averaged.
    print(rotation)
