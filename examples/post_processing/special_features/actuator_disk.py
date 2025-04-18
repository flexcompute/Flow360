import os

from pylab import show

import flow360 as fl
from flow360.examples import ActuatorDisk

ActuatorDisk.get_files()

project = fl.Project.from_volume_mesh(
    ActuatorDisk.mesh_filename,
    name="Actuator disk results from Python",
    length_unit="inch",
)

vm = project.volume_mesh

with fl.SI_unit_system:
    actuator_disk = fl.Cylinder(
        name="Actuator Cylinder",
        center=[0, 0, 0],
        axis=[-1, 0, 0],
        height=30 * fl.u.inch,
        outer_radius=150 * fl.u.inch,
    )
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=16286.016316209487 * fl.u.inch**2,
            moment_center=[450, 0, 0] * fl.u.inch,
            moment_length=[72, 1200, 1200] * fl.u.inch,
        ),
        operating_condition=fl.AerospaceCondition.from_mach(mach=0.04),
        time_stepping=fl.Steady(),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    kappa_MUSCL=0.33,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    update_jacobian_frequency=2,
                    equation_evaluation_frequency=1,
                ),
            ),
            fl.ActuatorDisk(
                name="ActuatorDisk",
                force_per_area=fl.ForcePerArea(
                    radius=[
                        6.541490006056935e-06,
                        1.176559660811629e01,
                        2.348576620230164e01,
                        3.502422774076318e01,
                        4.635372501514235e01,
                        5.740157480314961e01,
                        6.809963658388855e01,
                        7.837522713506966e01,
                        8.816929133858268e01,
                        9.741823137492429e01,
                        1.060675348273773e02,
                        1.140626892792247e02,
                        1.213537250151423e02,
                        1.278952150211993e02,
                        1.336508176862508e02,
                        1.385841913991520e02,
                        1.426589945487583e02,
                        1.458570563294973e02,
                        1.481511205330103e02,
                        1.495366444579043e02,
                        1.500000000000000e02,
                    ]
                    * fl.u.inch,
                    thrust=[
                        4.575739438020081686e-03,
                        5.344520067175258585e-03,
                        7.032044311381894543e-03,
                        8.710826002579061603e-03,
                        1.000951815124362203e-02,
                        1.093101144022547849e-02,
                        1.157119193478976620e-02,
                        1.201254985284840558e-02,
                        1.230866858381426607e-02,
                        1.248422468860116950e-02,
                        1.253851312261157827e-02,
                        1.244967750332181926e-02,
                        1.217471011028209080e-02,
                        1.166002755407952347e-02,
                        1.085486662369139835e-02,
                        9.723270044643288201e-03,
                        8.260302504752431441e-03,
                        6.493108221024028980e-03,
                        4.476116636387995722e-03,
                        2.283286929126045455e-03,
                        0.000000000000000000e00,
                    ]
                    * fl.u.psi,
                    circumferential=[
                        -5.531604620568226207e-12,
                        -2.000974664192053335e-03,
                        -2.952277367252184072e-03,
                        -3.088901336796338881e-03,
                        -2.903625904926704273e-03,
                        -2.643040675210779362e-03,
                        -2.392185737959950324e-03,
                        -2.173054219978886453e-03,
                        -1.986979038043627539e-03,
                        -1.827962066661991498e-03,
                        -1.688538971888151399e-03,
                        -1.560312378033051808e-03,
                        -1.435018201085241484e-03,
                        -1.304645611220716511e-03,
                        -1.162530024559274696e-03,
                        -1.004486082424148200e-03,
                        -8.291008930327343910e-04,
                        -6.375074356446677401e-04,
                        -4.326781146705185031e-04,
                        -2.186783262772103913e-04,
                        -0.000000000000000000e00,
                    ]
                    * fl.u.psi,
                ),
                volumes=actuator_disk,
            ),
            fl.Wall(surfaces=vm["fluid/body"]),
            fl.Freestream(surfaces=vm["fluid/farfield"]),
        ],
    )

case = project.run_case(params, "Actuator disk case from Python")


case.wait()

results = case.results

actuator_disk_non_dim = results.actuator_disks.as_dataframe()
print(actuator_disk_non_dim)

actuator_disk_non_dim.plot(
    x="pseudo_step",
    y=["Disk0_Power", "Disk0_Force", "Disk0_Moment"],
    xlim=(0, 200),
    xlabel="Pseudo Step",
    figsize=(10, 7),
    subplots=True,
    title="Actuator Disk non-dimensional",
)
show()

results.actuator_disks.to_base("SI")
actuator_disk_si = results.actuator_disks.as_dataframe()
print(actuator_disk_si)

actuator_disk_si.plot(
    x="pseudo_step",
    y=["Disk0_Power", "Disk0_Force", "Disk0_Moment"],
    xlim=(0, 200),
    xlabel="Pseudo Step",
    figsize=(10, 7),
    subplots=True,
    title="Actuator Disk scaled to SI",
)
show()

# download resuts:
results.set_destination(use_case_name=True)
results.download(actuator_disks=True, overwrite=True)

# save converted results to a new CSV file:
results.actuator_disks.to_file(os.path.join(case.name, "actuator_disk_in_SI.csv"))
