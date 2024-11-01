import os

import flow360.component.v1.modules as fl
import flow360.component.v1.units as u
from flow360.examples import OM6wing

here = os.path.dirname(os.path.abspath(__file__))

OM6wing.get_files()


class datafiles:
    output = os.path.join(here, "outputs.yaml")
    geometry = os.path.join(here, "geometry.yaml")
    boundaries = os.path.join(here, "boundaries.yaml")


# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# read simulation params from multiple files (.construct() is used to skip validation)
outputs = fl.Flow360Params.construct(datafiles.output)

with fl.SI_unit_system:
    params = fl.Flow360Params(
        geometry=fl.Geometry(datafiles.geometry),
        boundaries=fl.Boundaries(datafiles.boundaries),
        fluid_properties=fl.air,
        freestream=fl.FreestreamFromVelocity(velocity=286 * u.m / u.s, alpha=3.06),
        slice_output=outputs.slice_output,
        surface_output=outputs.surface_output,
        volume_output=outputs.volume_output,
    )


case = fl.Case.create("om6wing-from-yaml", params, volume_mesh_id=volume_mesh.id)
case = case.submit()
print(case)
