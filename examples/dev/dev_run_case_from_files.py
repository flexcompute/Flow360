import flow360 as fl
from flow360.examples import OM6wing

# Select the environment to which you want to submit your project
fl.Env.dev.active()  # Dev
# fl.Env.preprod.active() # Preprod
# fl.Env.uat.active() # UAT
# fl.Env.prod.active() # Prod (it is also the default environment)

OM6wing.get_files()

project = fl.Project.from_local_volume_mesh(
    OM6wing.mesh_filename, name="OM6wing Quick Start from Python"
)

vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0.0, 0.0, 0.0],
            moment_length=[1.47602, 0.801672958512342, 1.47602],
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=14.6e6, mach=0.84, project_length_unit=fl.u.m, alpha=3.06 * fl.u.deg
        ),
        time_stepping=fl.Steady(max_steps=500, CFL=fl.RampCFL(initial=5, final=200, ramp_steps=40)),
        models=[
            fl.Fluid(),
            fl.Wall(name="Wall", surfaces=[vm["1"]]),
            fl.SlipWall(name="SlipWall", surfaces=[vm["2"]]),
            fl.Freestream(name="Freestream", surfaces=[vm["3"]]),
        ],
        outputs=[
            fl.SurfaceOutput(output_fields=["primitiveVars", "Cp", "Cf"], surfaces=[vm["1"]]),
            fl.VolumeOutput(output_fields=["primitiveVars", "Mach"]),
        ],
    )

project.run_case(params, name="Case of OM6Wing Quick Start")
