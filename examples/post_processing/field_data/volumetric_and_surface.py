import os
import tarfile
import tempfile

import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(
    OM6wing.mesh_filename,
    name="Volumetric and surface results from Python",
)

vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0, 0, 0],
            moment_length=[1.47602, 0.801672958512342, 1.47602],
        ),
        operating_condition=fl.AerospaceCondition.from_mach_reynolds(
            reynolds=14.6e6, mach=0.84, alpha=3.06 * fl.u.deg, characteristic_length=fl.u.m
        ),
        time_stepping=fl.Steady(
            max_steps=500, CFL=fl.RampCFL(initial=5, final=200, ramp_steps=100)
        ),
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
            fl.VolumeOutput(output_fields=["primitiveVars", "Mach"]),
            fl.SurfaceOutput(surfaces=vm["1"], output_fields=["primitiveVars", "Cp", "Cf"]),
            fl.SliceOutput(
                output_fields=["Cp"],
                slices=[
                    fl.Slice(name="x0", normal=[1, 0, 0], origin=[0, 0, 0]),
                    fl.Slice(name="y1", normal=[0, 1, 0], origin=[2, 1, 0]),
                ],
            ),
        ],
    )

case = project.run_case(params, "Volumetric and surface results case from Python")


# wait until the case finishes execution
case.wait()

results = case.results

with tempfile.TemporaryDirectory() as temp_dir:
    # download slice and volume output files as tar.gz archives
    results.slices.download(os.path.join(temp_dir, "slices.tar.gz"), overwrite=True)
    results.volumes.download(os.path.join(temp_dir, "volumes.tar.gz"), overwrite=True)

    # slices.tar.gz, volumes.tar.gz
    print(os.listdir(temp_dir))

    # extract slices file
    file = tarfile.open(os.path.join(temp_dir, "slices.tar.gz"))
    file.extractall(os.path.join(temp_dir, "slices"))
    file.close()

    # contains plots for all slices in the specified format (tecplot)
    # slice_x1.szplt, slice_y1.szplt
    print(os.listdir(os.path.join(temp_dir, "slices")))

    # extract volumes file
    file = tarfile.open(os.path.join(temp_dir, "volumes.tar.gz"))
    file.extractall(os.path.join(temp_dir, "volumes"))
    file.close()

    # contains volume plots in the specified format (tecplot)
    # volume.szplt
    print(os.listdir(os.path.join(temp_dir, "volumes")))
