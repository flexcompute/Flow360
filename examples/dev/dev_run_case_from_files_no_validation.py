import os

import flow360.component.v1.modules as fl
from flow360.examples import OM6wing

fl.UserConfig.disable_validation()
OM6wing.get_files()


# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# # submit case using json file
here = os.path.dirname(os.path.abspath(__file__))
params = fl.UnvalidatedFlow360Params(
    os.path.join(here, "../../tests/data/case_params/incorrect.json")
)
case = fl.Case.create("OM6wing", params, volume_mesh.id, solver_version="release-23.2.1.0")
case = case.submit(force_submit=True)
print(case)
