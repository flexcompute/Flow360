from flow360.examples import OM6wing
from flow360.component.volume_mesh import VolumeMesh
from flow360.component.case import Case
from flow360.component.v1.flow360_params import Flow360Params

vm = VolumeMesh.copy_from_example("2ad77a88-1676-4f89-8652-13bd7e34f257")

params = Flow360Params(OM6wing.case_json)
case = Case.create("OM6wing", params, vm.id, solver_version="release-24.2")
case = case.submit()
print(case)
