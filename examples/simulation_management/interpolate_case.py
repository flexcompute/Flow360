import flow360 as fl

project = fl.Project.from_cloud("PROJECT_ID_HERE")

parent_case = fl.Case(id="PARENT_CASE_ID_HERE")

mesh = fl.VolumeMesh(id="MESH_ID_HERE")

project.run_case(params=parent_case.params, fork_from=parent_case, interpolate_to_mesh=mesh, name="Interpolated case")
