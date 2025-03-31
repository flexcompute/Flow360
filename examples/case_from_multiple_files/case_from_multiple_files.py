import os

import flow360 as fl
from flow360.examples import OM6wing

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

OM6wing.get_files()


class datafiles:
    boundary_wall = os.path.join(here, "models_wall.yaml")
    boundary_slipwall = os.path.join(here, "models_slipwall.yaml")
    bounrday_freestream = os.path.join(here, "models_freestream.yaml")
    surface_output = os.path.join(here, "outputs_surface.yaml")
    volume_output = os.path.join(here, "outputs_volume.yaml")
    operating_condition = os.path.join(here, "operating_condition.yaml")
    reference_geometry = os.path.join(here, "reference_geometry.yaml")
    time_stepping = os.path.join(here, "time_stepping.yaml")


project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="OM6Wing from Python")

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry.from_file(datafiles.reference_geometry),
        operating_condition=fl.AerospaceCondition.from_file(datafiles.operating_condition),
        time_stepping=fl.Steady.from_file(datafiles.time_stepping),
        models=[
            fl.Wall.from_file(datafiles.boundary_wall),
            fl.SlipWall.from_file(datafiles.boundary_slipwall),
            fl.Freestream.from_file(datafiles.bounrday_freestream),
        ],
        outputs=[
            fl.SurfaceOutput.from_file(datafiles.surface_output),
            fl.VolumeOutput.from_file(datafiles.volume_output),
        ],
    )

project.run_case(params, "OM6Wing case yaml from Python")
