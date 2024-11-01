import flow360.component.v1xxx as fl
from flow360.examples import OM6wing

OM6wing.get_files()

volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()


with fl.SI_unit_system:
    params = fl.Flow360Params(
        geometry=fl.Geometry(
            ref_area=1.15315084119231,
            moment_length=(1.47602, 0.801672958512342, 1.47602),
            mesh_unit="m",
        ),
        freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
        fluid_properties=fl.air,
        time_stepping=fl.SteadyTimeStepping(max_pseudo_steps=500),
        boundaries={
            "1": fl.NoSlipWall(name="wing"),
            "2": fl.SlipWall(name="symmetry"),
            "3": fl.FreestreamBoundary(name="freestream"),
        },
    )

case = volume_mesh.create_case("OM6wing", params)
case = case.submit()
