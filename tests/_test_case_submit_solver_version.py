import flow360 as fl
from flow360.examples import OM6wing

fl.UserConfig.set_profile("auto_test_1")

OM6wing.set_version("release-22.3.3.0")
OM6wing.get_files()


vm = fl.VolumeMesh.from_file(
    OM6wing.mesh_filename, name="OM6wing-mesh", solver_version="release-22.3.3.0"
).submit()
params = fl.Flow360Params(OM6wing.case_json)


case = vm.create_case(name="test-release-22.3.3.0", params=params).submit()
assert case.solver_version == "release-22.3.3.0"
case2 = case.fork(name="test-fork-release-22.3.3.0").submit()
assert case2.solver_version == "release-22.3.3.0"
assert case2.info.parent_id == case.id
case3 = case2.retry(name="test-retry-fork-release-22.3.3.0").submit()
assert case3.solver_version == "release-22.3.3.0"
assert case3.info.parent_id == case2.info.parent_id
case4 = case.retry(name="test-retry-release-23.1.1.0", solver_version="release-23.1.1.0").submit()
assert case4.solver_version == "release-23.1.1.0"
assert case4.info.parent_id == None

case = vm.create_case(
    name="test-release-23.2.1.0", params=params, solver_version="release-23.2.1.0"
).submit()
assert case.solver_version == "release-23.2.1.0"
case2 = case.fork(name="test-fork-release-23.2.1.0").submit()
assert case2.solver_version == "release-23.2.1.0"
assert case2.info.parent_id == case.id
case3 = case.retry(name="test-retry-release-23.1.1.0", solver_version="release-23.1.1.0").submit()
assert case3.solver_version == "release-23.1.1.0"
assert case3.info.parent_id == None
case4 = case2.retry(name="test-retry-fork-release-23.2.1.0").submit()
assert case4.solver_version == "release-23.2.1.0"
assert case4.info.parent_id == case.id

# this is not allowed to change solver version from parent
case4 = case2.retry(
    name="test-retry-fork-release-23.1.1.0", solver_version="release-23.1.1.0"
).submit()
assert case4.solver_version == "release-23.1.1.0"
