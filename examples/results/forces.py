import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()

# submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = volume_mesh.create_case("OM6wing-example", params)
case = case.submit()

# wait until the case finishes execution
case.wait()

results = case.results

print(results.total_forces.as_dataframe())
# >>>
#     physical_step  pseudo_step        CL        CD       CFx       CFy       CFz       CMx       CMy  ... CMzSkinFriction
# 0               0            0  0.070379  0.044693  0.040872 -0.030555  0.072664  0.033416 -0.061036  ...        0.000000
# 1               0           10  0.230288  0.255790  0.243132 -0.077975  0.243614  0.111144 -0.233192  ...       -0.043915
# 2               0           20  0.241752  0.239167  0.225921 -0.075197  0.254174  0.116058 -0.244510  ...       -0.038846
# 3               0           30  0.239568  0.226566  0.213454 -0.074483  0.251321  0.114734 -0.241601  ...       -0.033753
# 4               0           40  0.239892  0.216702  0.203587 -0.074562  0.251118  0.114637 -0.241349  ...       -0.029254
# 5               0           50  0.240130  0.207138  0.194024 -0.074126  0.250845  0.114500 -0.241074  ...       -0.025312
# 6               0           60  0.240189  0.197541  0.184438 -0.072966  0.250392  0.114257 -0.240625  ...       -0.021926


print(results.surface_forces.as_dataframe())
# >>>
#   physical_step  pseudo_step   wing_CL   wing_CD  wing_CFx  wing_CFy  wing_CFz  wing_CMx  ...
# 0               0            0  0.070379  0.044693  0.040872 -0.030555  0.072664  0.033416  ...
# 1               0           10  0.230288  0.255790  0.243132 -0.077975  0.243614  0.111144  ...
# 2               0           20  0.241752  0.239167  0.225921 -0.075197  0.254174  0.116058  ...


# force distribution is post-processing. We need to wait for results.
results.force_distribution.wait()
print(results.force_distribution.as_dataframe())
# >>>
#             Y  wing_CFx_per_span  wing_CFz_per_span  wing_CMy_per_span
# 0    0.000000           0.000000           0.000000           0.000000
# 1    0.005024           0.031981           0.219291          -0.100513
# 2    0.010049           0.031490           0.219799          -0.101065
# 3    0.015073           0.031002           0.220269          -0.101596
# 4    0.020098           0.030516           0.220701          -0.102106
# ..        ...                ...                ...                ...
# 295  1.482194          -0.011669           0.029717          -0.032998
# 296  1.487218          -0.011045           0.023575          -0.028619
# 297  1.492243          -0.005613           0.017790          -0.022960
# 298  1.497267          -0.000027           0.011064          -0.014674
# 299  1.502291           0.000000           0.000000           0.000000


results.set_destination(use_case_name=True)
results.set_downloader(total_forces=True, surface_forces=True, force_distribution=True)
results.download()
