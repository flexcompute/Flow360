import os

import flow360 as fl
from flow360.examples import ActuatorDisk

ActuatorDisk.get_files()

# # submit mesh
# volume_mesh = fl.VolumeMesh.from_file(ActuatorDisk.mesh_filename, name="ActuatorDisk-mesh")
# volume_mesh = volume_mesh.submit()

# # # submit case using json file
# params = fl.Flow360Params(ActuatorDisk.case_json)
# case = volume_mesh.create_case("ActuatorDisk-example", params)
# case = case.submit()

# case.wait()

project = fl.Project.from_file(
    files=fl.VolumeMeshFile(ActuatorDisk.mesh_filename),
    name="Actuator disk case from Python"
)

vm = project.volume_mesh

# with fl.SI_unit_system:
#     actuator_disk = fl.Cylinder(
#         center=[0, 0, 0],
#         axis=[-1, 0, 0],
#         height=30,
#         outer_radius=150
#     )
#     params = fl.SimulationParams(
#         reference_geometry=fl.ReferenceGeometry(
#             area=16286.016316209487,
#             moment_center=[450, 0, 0],
#             moment_length=[72, 1200, 1200]
#         ),
#         operating_condition=fl.AerospaceCondition.from_mach(mach=0.04),
#         time_stepping=fl.Steady(
#             max_steps=200,
#             CFL=fl.RampCFL(
#                 initial=1,
#                 final=200,
#                 ramp_steps=200
#             )
#         ),
#         models=[
#             fl.ActuatorDisk(
#                 force_per_area=fl.ForcePerArea(
#                     radius=[
#                         -6.541490006056935e-06,1.176559660811629e+01,2.348576620230164e+01,3.502422774076318e+01,4.635372501514235e+01,5.740157480314961e+01,6.809963658388855e+01,7.837522713506966e+01,8.816929133858268e+01,9.741823137492429e+01,1.060675348273773e+02,1.140626892792247e+02,1.213537250151423e+02,1.278952150211993e+02,1.336508176862508e+02,1.385841913991520e+02,1.426589945487583e+02,1.458570563294973e+02,1.481511205330103e+02,1.495366444579043e+02,1.500000000000000e+02
#                     ],
#                     thrust=[
#                         4.575739438020081686e-03, 5.344520067175258585e-03, 7.032044311381894543e-03, 8.710826002579061603e-03, 1.000951815124362203e-02, 1.093101144022547849e-02, 1.157119193478976620e-02, 1.201254985284840558e-02, 1.230866858381426607e-02, 1.248422468860116950e-02, 1.253851312261157827e-02, 1.244967750332181926e-02, 1.217471011028209080e-02, 1.166002755407952347e-02, 1.085486662369139835e-02, 9.723270044643288201e-03, 8.260302504752431441e-03, 6.493108221024028980e-03, 4.476116636387995722e-03, 2.283286929126045455e-03, 0.000000000000000000e+00
#                     ],
#                     circumferential=[
#                         -5.531604620568226207e-12, -2.000974664192053335e-03, -2.952277367252184072e-03, -3.088901336796338881e-03, -2.903625904926704273e-03, -2.643040675210779362e-03, -2.392185737959950324e-03, -2.173054219978886453e-03, -1.986979038043627539e-03, -1.827962066661991498e-03, -1.688538971888151399e-03, -1.560312378033051808e-03, -1.435018201085241484e-03, -1.304645611220716511e-03, -1.162530024559274696e-03, -1.004486082424148200e-03, -8.291008930327343910e-04, -6.375074356446677401e-04, -4.326781146705185031e-04, -2.186783262772103913e-04, -0.000000000000000000e+00
#                     ]
#                 )
#             )
#         ]
#     )











# results = case.results

# # set values needed for units conversion:
# case.params.geometry.mesh_unit = 1 * u.m
# case.params.fluid_properties = fl.air

# print(results.actuator_disks)
# # >>>
# #     physical_step  pseudo_step  Disk0_Power  Disk0_Force  Disk0_Moment
# # 0               0            0    30.062549   751.563715  10537.291912
# # 1               0           10    31.667677   751.563715  10537.291912
# # 2               0           20    33.258442   751.563715  10537.291912
# # 3               0           30    34.296091   751.563715  10537.291912
# # 4               0           40    34.762001   751.563715  10537.291912
# # 5               0           50    35.396422   751.563715  10537.291912
# # 6               0           60    37.026789   751.563715  10537.291912
# # 7               0           70    40.024032   751.563715  10537.291912


# # convert results to SI system:
# results.actuator_disks.to_base("SI")
# print(results.actuator_disks)
# # >>>
# #     physical_step  pseudo_step   Disk0_Power   Disk0_Force  Disk0_Moment    PowerUnits ForceUnits   MomentUnits
# # 0               0            0  1.451192e+09  1.066131e+08  1.451192e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 1               0           10  1.528675e+09  1.066131e+08  1.528675e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 2               0           20  1.605465e+09  1.066131e+08  1.605465e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 3               0           30  1.655555e+09  1.066131e+08  1.655555e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 4               0           40  1.678046e+09  1.066131e+08  1.678046e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 5               0           50  1.708671e+09  1.066131e+08  1.708671e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 6               0           60  1.787372e+09  1.066131e+08  1.787372e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2
# # 7               0           70  1.932056e+09  1.066131e+08  1.932056e+09  kg*m**2/s**3  kg*m/s**2  kg*m**2/s**2

# # download resuts:
# results.set_destination(use_case_name=True)
# results.download(actuator_disks=True, overwrite=True)

# # save converted results to a new CSV file:
# results.actuator_disks.to_file(os.path.join(case.name, "actuator_disk_in_SI.csv"))
