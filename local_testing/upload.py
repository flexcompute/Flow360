import flow360 as fl


# project = fl.Project.from_geometry(
#     "snappyTest.stl",
#     name="snappyTest",
#     solver_version="snappyHex-25.7.0"
# )

project = fl.Project.from_cloud(project_id="prj-0ae76f32-1c87-4c2d-b084-2d18bec68f17")

# print(project.geometry)