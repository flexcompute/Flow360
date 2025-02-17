import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()


project = fl.Project.from_file(
    files=fl.VolumeMeshFile(OM6wing.mesh_filename),
    name="OM6Wing json input from Python"
)

params = fl.SimulationParams.from_file("data/om6wing_params.yaml")

project.run_case(params, name="OM6Wing json input case from Python")