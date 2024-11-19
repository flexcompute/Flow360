import flow360 as fl
import math

fl.Env.dev.active()

parent_project = fl.Project.from_file(
    "/local_data/ben/testcases/localTests/restartInterpolation/cartesian_2d_mesh.cgns",
    name="ParentCaseProject",
)
volume_mesh = parent_project.volume_mesh


##### Setup the param #####
U_inf = 0.5
P_inf = 1.0 / 1.4
T_inf = 1.0
Radius = 1
Cp = 1.0 / (1.4 - 1)
Rgas = 1.0 / 1.4
Xc = 0.0
Yc = 0.0
Beta = 0.2
Alpha = 45.0 * math.pi / 180.0
mesh_unit = 1 * fl.u.m

time_step_size = 0.1414213562373095 * mesh_unit / fl.Air().get_speed_of_sound(288.15 * fl.u.K)
num_time_steps = 300

densityField = f"{P_inf}/({Rgas}*{T_inf})*pow(({T_inf}-({U_inf}*{U_inf}*{Beta}*{Beta})/(2*{Cp})*exp(-(pow(x-{Xc},2)+pow(y-{Yc},2))/({Radius}*{Radius})))/{T_inf},1/(1.4-1.))"
uField = f"{U_inf}*{Beta}*exp(-0.5*(pow(x-{Xc}, 2)+pow(y-{Yc},2))/({Radius}*{Radius}))/{Radius}*(-1*(y-{Yc})) + cos({Alpha})*{U_inf}"
vField = f"{U_inf}*{Beta}*exp(-0.5*(pow(x-{Xc},2)+pow(y-{Yc},2))/({Radius}*{Radius}))/{Radius}*(x-{Xc}) + sin({Alpha})*{U_inf}"
wField = "0"
pressureField = f"{P_inf}/({Rgas}*{T_inf})*pow(({T_inf}-({U_inf}*{U_inf}*{Beta}*{Beta})/(2*{Cp})*exp(-(pow(x-{Xc}, 2)+pow(y-{Yc}, 2))/({Radius}*{Radius})))/{T_inf},1/(1.4-1.)) * {Rgas} * ({T_inf}-({U_inf}*{U_inf}*{Beta}*{Beta})/(2*{Cp})*exp(-(pow(x-{Xc}, 2)+pow(y-{Yc}, 2))/({Radius}*{Radius})))"

with fl.SI_unit_system:

    params = fl.SimulationParams(
        operating_condition=fl.AerospaceCondition.from_mach(
            mach=0.5,
            thermal_state=fl.ThermalState(
                material=fl.Air(
                    dynamic_viscosity=fl.Sutherland(
                        reference_viscosity=1e-100,
                        effective_temperature=110.4 * fl.u.K,
                        reference_temperature=288.15 * fl.u.K,
                    )
                )
            ),
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    kappa_MUSCL=-1.0,
                ),
                turbulence_model_solver=fl.NoneSolver(),
                initial_condition=fl.NavierStokesInitialCondition(
                    rho=densityField, u=uField, v=vField, w=wField, p=pressureField
                ),
            ),
            fl.SlipWall(surfaces=[volume_mesh["VOLUME/BACK"], volume_mesh["VOLUME/FRONT"]]),
            fl.Periodic(
                surface_pairs=[
                    (volume_mesh["VOLUME/RIGHT"], volume_mesh["VOLUME/LEFT"]),
                    (volume_mesh["VOLUME/BOTTOM"], volume_mesh["VOLUME/TOP"]),
                ],
                spec=fl.Translational(),
            ),
        ],
        time_stepping=fl.Unsteady(
            CFL=fl.RampCFL(initial=100, final=10000, ramp_steps=5),
            step_size=time_step_size,
            steps=num_time_steps,
            max_pseudo_steps=1,
        ),
    )

parent_project.run_case(params=params, name="ParentCase")

mesh_provider_project = fl.Project.from_file(
    "/local_data/ben/testcases/localTests/restartInterpolation/unstructured_2d_mesh.cgns",
    name="MeshProviderProject",
)

mesh_provider_project.volume_mesh.wait()  # For the volume mesh to be ready

parent_project.run_case(
    params=params,
    name="ForkButWithDifferentMesh",
    fork_from=parent_project.case,
    fork_with_mesh=mesh_provider_project.volume_mesh,
)
