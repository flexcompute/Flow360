import flow360 as fl
import numpy as np

project = fl.Project.from_geometry("geometry.csm", name="Center of pressure")
geometry = project.geometry
geometry.show_available_groupings(verbose_mode=True)  # lists the selectable names

with (
    fl.create_draft(
        new_run_from=geometry, face_grouping="groupName", edge_grouping="edgeName"
    ) as draft,
    fl.SI_unit_system,
):
    # One surface, a group of them, or draft.surfaces["*"] for every wall.
    surfaces = [draft.surfaces["wing"], draft.surfaces["flap"]]

    # --- DURING THE RUN: live traces in the WebUI under Analysis -> Monitor ---
    # The solver works in grid units, so these traces are in grid units too. eps
    # floors the denominator: on the first step the force is still zero, and
    # without it the division would give a NaN that every later step inherits.
    cop_monitor = fl.UserDefinedDynamic(
        name="centerOfPressure",
        input_vars=["forceX", "forceY", "forceZ", "momentX", "momentY", "momentZ"],
        constants={"eps": 1e-12},
        state_vars_initial_value=["0.0", "0.0", "0.0", "0.0"],
        update_law=[
            # state[0] -> x_cp
            "momentCenterX + (forceY * momentZ - forceZ * momentY)"
            " / max(forceX * forceX + forceY * forceY + forceZ * forceZ, eps);",
            # state[1] -> y_cp
            "momentCenterY + (forceZ * momentX - forceX * momentZ)"
            " / max(forceX * forceX + forceY * forceY + forceZ * forceZ, eps);",
            # state[2] -> z_cp
            "momentCenterZ + (forceX * momentY - forceY * momentX)"
            " / max(forceX * forceX + forceY * forceY + forceZ * forceZ, eps);",
            # state[3] -> couple_arm
            "abs(forceX * momentX + forceY * momentY + forceZ * momentZ)"
            " / max(forceX * forceX + forceY * forceY + forceZ * forceZ, eps);",
        ],
        input_boundary_patches=surfaces,  # inputs are summed over these surfaces
    )

    # moment_center defaults to None, so set the reference geometry explicitly.
    # The block after the run reads it back from the case.
    params = fl.SimulationParams(
        ...,
        reference_geometry=fl.ReferenceGeometry(
            moment_center=(0, 0, 0),
            moment_length=(1, 1, 1),
            area=1,
        ),
        user_defined_dynamics=[cop_monitor],
    )

    # The draft has to stay active while the run is submitted.
    case = project.run_case(params, name="Center of pressure")

case.wait()

# The same data as the WebUI traces, one row per output step, in grid units.
print(case.results.user_defined_dynamics["centerOfPressure"].as_dataframe().tail())

# --- AFTER THE RUN: the single converged answer, converted to metres ---
reference_geometry = case.params.reference_geometry
moment_center = reference_geometry.moment_center.to(fl.u.m).value
moment_length = np.broadcast_to(
    np.atleast_1d(reference_geometry.moment_length.to(fl.u.m).value), 3
)

# filter() re-sums the total* columns over the selected surfaces. Boundaries carry a
# zone prefix, as in "fluid/wing", so the patterns are wildcards; print
# surface_forces.entities for the actual names. exclude= is available too.
surface_forces = case.results.surface_forces
surface_forces.filter(include=["*wing*", "*flap*"])

# Average first, then reduce: the mean of a ratio is not the ratio of the means.
coefficients = surface_forces.get_averages(0.1)
CF = np.array([coefficients[f"totalCF{i}"] for i in "xyz"])
CM = np.array([coefficients[f"totalCM{i}"] for i in "xyz"]) * moment_length

# the point of the line of action closest to the moment center
center_of_pressure = moment_center + np.cross(CF, CM) / (CF @ CF)
# the part of the moment that no point can cancel, expressed as a length
couple_arm = abs(CM @ CF) / (CF @ CF)
# slide along the line into z = z_mc for the conventional chordwise position
in_plane = center_of_pressure + CF * (moment_center[2] - center_of_pressure[2]) / CF[2]

print(f"center of pressure  : {center_of_pressure} m")
print(f"same line, at z_mc  : {in_plane} m")
print(f"residual couple arm : {couple_arm:.4f} m")
