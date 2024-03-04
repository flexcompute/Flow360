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
    monitor = results.monitors.get_monitor_by_name(name)
    # >>>
    # physical_step  pseudo_step  ...  Group1_Point4_mut  Group1_Point4_Cp
    # 0               0            0  ...       1.210739e-08          0.000000
    # 1               0           10  ...       1.210737e-08         -0.000005
    # 2               0           20  ...       1.212245e-08         -0.000580
    # 3               0           30  ...       1.250081e-08          0.000615
    # 4               0           40  ...       1.366265e-08         -0.003011
    # etc.
    print(monitor)

    monitor.download(to_folder=case.name)
