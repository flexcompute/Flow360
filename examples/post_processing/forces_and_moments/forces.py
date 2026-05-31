from pylab import show

import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="Forces results from Python")

vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0, 0, 0],
            moment_length=[1.47602, 0.801672958512342, 1.47602],
        ),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=286, alpha=3.06 * fl.u.deg),
        time_stepping=fl.Steady(max_steps=500),
        models=[
            fl.Wall(surfaces=vm["1"]),
            fl.SlipWall(surfaces=vm["2"]),
            fl.Freestream(surfaces=vm["3"]),
        ],
    )

case = project.run_case(params, "Forces results case from Python")


# wait until the case finishes execution
case.wait()

results = case.results

total_forces = results.total_forces.as_dataframe()
print(total_forces)

total_forces.plot(
    x="pseudo_step",
    y=["CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"],
    xlabel="Pseudo Step",
    xlim=(0, None),
    figsize=(10, 7),
    title="Total forces",
)
show()

surface_forces = results.surface_forces.as_dataframe()
print(surface_forces)

results.set_destination(use_case_name=True)
results.download(total_forces=True, surface_forces=True)
