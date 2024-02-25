import flow360 as fl
from flow360.examples import UserDefinedDynamics

UserDefinedDynamics.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(UserDefinedDynamics.mesh_filename, name="UDD-mesh")
volume_mesh = volume_mesh.submit()

# submit case using json file
params = fl.Flow360Params(UserDefinedDynamics.case_json)
case = volume_mesh.create_case("UDD-example", params)
case = case.submit()

# wait until the case finishes execution
case.wait()

results = case.results

udd = results.user_defined_dynamics

for name in udd.udd_names:
    result = udd.get_udd_by_name(name)
    #      physical_step  pseudo_step        CL  state[0]  state[1]  alphaAngle
    # 0                0            0  0.000000  3.060000  0.000000    3.060000
    # 1                0           10  0.143092  3.060000  0.000000    3.060000
    # 2                0           20  0.239919  3.060000  0.000000    3.060000
    # 3                0           30  0.227201  3.060000  0.000000    3.060000
    # 4                0           40  0.222182  3.060000  0.000000    3.060000
    # etc.
    print(result.as_dataframe())
