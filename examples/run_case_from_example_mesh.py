import flow360.component.v1.modules as fl
from flow360.examples import OM6wing

vm = fl.VolumeMesh.copy_from_example("2ad77a88-1676-4f89-8652-13bd7e34f257")

params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing", params, vm.id, solver_version="release-24.2")
case = case.submit()
print(case)
