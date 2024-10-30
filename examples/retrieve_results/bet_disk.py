import os

import flow360.component.v1 as fl
import flow360.component.v1.units as u
from flow360.examples import BETDisk

BETDisk.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(BETDisk.mesh_filename, name="BETDisk-mesh")
volume_mesh = volume_mesh.submit()

# # submit case using json file
params = fl.Flow360Params(BETDisk.case_json)
case = volume_mesh.create_case("BETDisk-example", params)
case = case.submit()


case.wait()


results = case.results

# set values needed for units conversion:
case.params.geometry.mesh_unit = 1 * u.m
case.params.fluid_properties = fl.air

print(results.bet_forces)
# >>>
#     physical_step  pseudo_step  Disk0_Force_x  Disk0_Force_y  Disk0_Force_z  Disk0_Moment_x  ...
# 0               0            0   -1397.096153       0.010873      -0.000516   162623.186588  ...
# 1               0           10   -1310.554720       0.005608      -0.000557   159995.121031  ...
# 2               0           20   -1214.661435       0.001554      -0.000484   156872.026442  ...
# 3               0           30   -1152.224164      -0.001009      -0.000956   153785.239849  ...
# 4               0           40   -1120.710025      -0.006711      -0.001384   150297.612151  ...
# 5               0           50   -1070.491683      -0.033287      -0.005191   145265.442238  ...
# 6               0           60    -954.888260      -0.051523      -0.012721   137884.126198  ...
# 7               0           70    -764.398174      -0.016856       0.008907   130052.235155  ...


# convert results to SI system:
results.bet_forces.to_base("SI")
print(results.bet_forces)
# >>>
#     physical_step  pseudo_step  Disk0_Force_x  Disk0_Force_y  Disk0_Force_z  Disk0_Moment_x  ...  ForceUnits   MomentUnits
# 0               0            0  -1.981851e+08    1542.442287     -73.192214    2.306891e+10  ...   kg*m/s**2  kg*m**2/s**2
# 1               0           10  -1.859088e+08     795.544663     -78.962361    2.269611e+10  ...   kg*m/s**2  kg*m**2/s**2
# 2               0           20  -1.723058e+08     220.437099     -68.694195    2.225308e+10  ...   kg*m/s**2  kg*m**2/s**2
# 3               0           30  -1.634488e+08    -143.185024    -135.674806    2.181521e+10  ...   kg*m/s**2  kg*m**2/s**2
# 4               0           40  -1.589783e+08    -951.938460    -196.264715    2.132047e+10  ...   kg*m/s**2  kg*m**2/s**2
# 5               0           50  -1.518546e+08   -4721.975263    -736.389482    2.060663e+10  ...   kg*m/s**2  kg*m**2/s**2
# 6               0           60  -1.354557e+08   -7308.840572   -1804.542546    1.955955e+10  ...   kg*m/s**2  kg*m**2/s**2
# 7               0           70  -1.084337e+08   -2391.134119    1263.499262    1.844856e+10  ...   kg*m/s**2  kg*m**2/s**2


# download resuts:
results.set_destination(use_case_name=True)
results.download(bet_forces=True, overwrite=True)

# save converted results to a new CSV file:
results.bet_forces.to_file(os.path.join(case.name, "bet_forces_in_SI.csv"))
