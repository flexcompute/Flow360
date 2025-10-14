"""
Test specifying velocity direction for Inflow boundary with MassFlowRate.
The Inflow has non-X normal component but the velocity direction is (1, 0, 0)
Therefore the specified velocity direction should contain X only component at inlet.
"""

import flow360 as fl

fl.Env.dev.active()

if __name__ == "__main__":
    #meshFile = "cartesian_2d_mesh.oblique.cgns"
    #project = fl.Project.from_volume_mesh(meshFile, name="Test Imported Surface Output", solver_version="release-25.7.1")
    #vm = project.volume_mesh
    #params = createBaseParams(vm)
    #case = project.run_case(params, "Run_imported_surface_output")

    project = fl.Project.from_cloud("prj-59dd9622-fc8e-4324-bdc8-0e285b7acc05")
    parent_case = fl.Case(id="case-1cfa38d7-ad69-4c0b-93ff-aa54f2304415")
    params = parent_case.params

    project.run_case(params=params, fork_from=parent_case, name="fork via pythonUI")

