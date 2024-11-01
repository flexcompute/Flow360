import os

import flow360.units as u
import flow360.v1 as fl
from flow360.examples import ActuatorDisk

ActuatorDisk.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(ActuatorDisk.mesh_filename, name="ActuatorDisk-mesh")
volume_mesh = volume_mesh.submit()

# # submit case using json file
params = fl.Flow360Params(ActuatorDisk.case_json)
case = volume_mesh.create_case("ActuatorDisk-example", params)
case = case.submit()

case.wait()


results = case.results

# set values needed for units conversion:
case.params.geometry.mesh_unit = 1 * u.m
case.params.fluid_properties = fl.air

print(results.actuator_disks)
# >>>
#     physical_step  pseudo_step  Disk0_Power  Disk0_Force  Disk0_Moment
# 0               0            0    30.062549   751.563715  10537.291912
# 1               0           10    31.667677   751.563715  10537.291912
# 2               0           20    33.258442   751.563715  10537.291912
# 3               0           30    34.296091   751.563715  10537.291912
# 4               0           40    34.762001   751.563715  10537.291912
# 5               0           50    35.396422   751.563715  10537.291912
# 6               0           60    37.026789   751.563715  10537.291912
# 7               0           70    40.024032   751.563715  10537.291912


# convert results to SI system:
results.actuator_disks.to_base("SI")
print(results.actuator_disks)
# >>>
#     physical_step  pseudo_step   Disk0_Power   Disk0_Force  Disk0_Moment    PowerUnits ForceUnits   MomentUnits
# 0               0            0  1.451192e+09  1.066131e+08  1.451192e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 1               0           10  1.528675e+09  1.066131e+08  1.528675e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 2               0           20  1.605465e+09  1.066131e+08  1.605465e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 3               0           30  1.655555e+09  1.066131e+08  1.655555e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 4               0           40  1.678046e+09  1.066131e+08  1.678046e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 5               0           50  1.708671e+09  1.066131e+08  1.708671e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 6               0           60  1.787372e+09  1.066131e+08  1.787372e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# 7               0           70  1.932056e+09  1.066131e+08  1.932056e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2


# download resuts:
results.set_destination(use_case_name=True)
results.download(actuator_disks=True, overwrite=True)

# save converted results to a new CSV file:
results.actuator_disks.to_file(os.path.join(case.name, "actuator_disk_in_SI.csv"))
