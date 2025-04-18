import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="Monitors results from Python")

vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0, 0, 0],
            moment_length=[1.47602, 0.801672958512342, 1.47602],
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=14.6e6, mach=0.84, alpha=3.06 * fl.u.deg, project_length_unit=fl.u.m
        ),
        time_stepping=fl.Steady(max_steps=500),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
                turbulence_model_solver=fl.SpalartAllmaras(absolute_tolerance=1e-8),
            ),
            fl.Wall(surfaces=vm["1"]),
            fl.SlipWall(surfaces=vm["2"]),
            fl.Freestream(surfaces=vm["3"]),
        ],
        outputs=[
            fl.ProbeOutput(
                output_fields=["primitiveVars", "vorticity", "T", "s", "Cp", "mut"],
                probe_points=[
                    fl.Point(name="Probe1", location=[0.12, 0.34, 0.262] * fl.u.m),
                    fl.Point(name="Probe2", location=[2, 0.01, 0.03] * fl.u.m),
                    fl.Point(name="Probe3", location=[3, 0.01, 0.04] * fl.u.m),
                    fl.Point(name="Probe4", location=[4, 0.01, 0.04] * fl.u.m),
                ],
            )
        ],
    )

case = project.run_case(params, "Monitors results case from Python")


# wait until the case finishes execution
case.wait()

results = case.results

for name in results.monitors.monitor_names:
    monitor = results.monitors.get_monitor_by_name(name)
    print(monitor.as_dataframe())
    monitor.download(to_folder=case.name)
