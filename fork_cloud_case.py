import flow360 as fl

# fl.Env.preprod.active()


case = fl.Case.from_cloud("case-bb539f75-9e12-4c8c-a3c6-3d5e596d5e4a")
# project = fl.Project.from_cloud("prj-d5d1b4c9-f129-4fdf-962b-46f02527cfb5", new_run_from=case)
project = fl.Project.from_cloud("prj-d5d1b4c9-f129-4fdf-962b-46f02527cfb5")


param: fl.SimulationParams = case.params

# param.time_stepping.CFL.convergence_limiting_factor = 0.9
# param.models[0].private_attribute_dict = {"wallModelType": "InnerLayer"}
# param.models[0].navier_stokes_solver.private_attribute_dict = {"typeName": "CompressibleIsentropic"}
param.models[0].turbulence_model_solver.private_attribute_dict = {
    "debugPoint": [-1.027079e05, +0.000000e00, +1.763426e06]
}
# param.models[0].wall_model_type = "InnerLayer"
# param.outputs[0].output_format = "paraview"
# param.outputs[1].output_format = "paraview"
# param.outputs[0].output_fields = ["Cf", "Cp", "yPlus", "residualNavierStokes", "residualTurbulence"]
# param.time_stepping.steps = 150
# param.outputs.append(
#     # fl.SliceOutput(
#     #     slices=[
#     #         fl.Slice(
#     #             name="slice1",
#     #             normal=[0.0, 1.0, 0.0],
#     #             origin=[1.316111e+00, 3.344018e+00, 8.446533e-01] * fl.u.m
#     #         )
#     #     ],
#     #     output_fields=["residualNavierStokes", "residualTurbulence", "mutRatio", "mut", "primitiveVars"],
#     #     output_format="paraview",
#     #     frequency=1,
#     # )
#     fl.SliceOutput(
#         slices=[
#             fl.Slice(
#                 name="slice1",
#                 normal=[0.0, 0.0, 1.0],
#                 origin=[2.7, 2.65, 1.0] * fl.u.m
#             )
#         ],
#         output_fields=["residualNavierStokes", "residualTurbulence", "mutRatio", "mut", "primitiveVars"],
#         output_format="paraview",
#         frequency=5,
#     )
# )
# param.models[0].turbulence_model_solver.order_of_accuracy = 1
# param.models[0].turbulence_model_solver.CFL_multiplier = 0.01
# param.models[0].turbulence_model_solver.equation_evaluation_frequency = 1
# param.models[0].turbulence_model_solver.linear_solver.max_iterations = 80
# param.time_stepping.step_size = 2.5e-4 * fl.u.s
# param.time_stepping.CFL.max = 100
# param.time_stepping.max_pseudo_steps = 100
# param.models[0].private_attribute_dict = {"debugType": "maxRes"}
# print(param.models)
#
project.run_case(params=param, name="elysian_debug_SA", solver_version="masspi-25.6.1")
