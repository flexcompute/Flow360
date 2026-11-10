import flow360 as fl

my_project = fl.Project.from_cloud("PROJECT_ID_HERE")

with fl.SI_unit_system:
    params = fl.SimulationParams(...)

my_project.run_case(params)
