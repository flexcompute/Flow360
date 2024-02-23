import flow360 as fl
from flow360.examples import MonitorsAndSlices

MonitorsAndSlices.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(MonitorsAndSlices.mesh_filename, name="Monitors-mesh")
volume_mesh = volume_mesh.submit()

# submit case using json file
params = fl.Flow360Params(MonitorsAndSlices.case_json)
case = volume_mesh.create_case("Monitors-example", params)
case = case.submit()

# wait until the case finishes execution
case.wait()

results = case.results

for name in results.monitors.monitor_names:
    result = results.monitors.get_monitor_by_name(name)
    # >>>
    # physical_step  pseudo_step  ...  Group1_Point4_mut  Group1_Point4_Cp
    # 0               0            0  ...       1.210739e-08          0.000000
    # 1               0           10  ...       1.210737e-08         -0.000005
    # 2               0           20  ...       1.212245e-08         -0.000580
    # 3               0           30  ...       1.250081e-08          0.000615
    # 4               0           40  ...       1.366265e-08         -0.003011
    # 5               0           50  ...       1.446943e-08          0.002356
    # 6               0           60  ...       1.547504e-08          0.003438
    # 7               0           70  ...       1.563407e-08          0.002713
    # 8               0           80  ...       1.559043e-08          0.000024
    # 9               0           90  ...       1.566935e-08          0.002605
    # 10              0          100  ...       1.615775e-08          0.001711
    # etc.
    print(result.as_dataframe())
