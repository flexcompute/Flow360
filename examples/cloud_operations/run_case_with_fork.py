import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="Forking cases from Python")
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
        outputs=[
            fl.SurfaceOutput(output_fields=["primitiveVars", "Cp", "Cf"], surfaces=[vm["1"]]),
            fl.VolumeOutput(output_fields=["primitiveVars", "Mach"]),
        ],
    )

case = project.run_case(params, "OM6Wing-default-0")

# fork a case
case.params.time_stepping.max_steps = 300
case_fork_1 = project.run_case(case.params, "OM6Wing-fork-1", fork_from=case)

# create another fork
case_fork_1.params.time_stepping.max_steps = 200
case_fork_2 = project.run_case(case_fork_1.params, "OM6Wing-fork-2", fork_from=case_fork_1)

# create fork by providing parent case id:
case_fork = project.run_case(params, "OM6Wing-fork-1-0", fork_from=case)
