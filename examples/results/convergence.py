import flow360 as fl
from flow360.examples import Convergence

Convergence.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(Convergence.mesh_filename, name="diverging-mesh")
volume_mesh = volume_mesh.submit()

# submit case using json file
params = fl.Flow360Params(Convergence.case_json)
case = volume_mesh.create_case("diverging-example", params)
case = case.submit()

# wait until the case finishes execution
case.wait()

results = case.results

# nonlinear residuals contain convergence information
#     physical_step  pseudo_step    0_cont  ...    3_momz   4_energ   5_nuHat
# 0               0            0  0.003167  ...  0.003540  0.009037  0.002040
# 1               0           10  0.001826  ...  0.001106  0.005233  0.000549
# 2               0           20  0.001690  ...  0.001118  0.004845  0.000369
# 3               0           30  0.001623  ...  0.001118  0.004662  0.000298
# 4               0           40  0.001538  ...  0.001077  0.004445  0.000265
# ..            ...          ...       ...  ...       ...       ...       ...
# 69              0          690  0.000034  ...  0.000020  0.000099  0.000700
# 70              0          700  0.000043  ...  0.000025  0.000121  0.000740
# 71              0          710  0.000042  ...  0.000021  0.000117  0.000712
# 72              0          720  0.000030  ...  0.000018  0.000084  0.000718
# 73              0          725       NaN  ...       NaN       NaN  0.002658
print(results.nonlinear_residuals.as_dataframe())
