import flow360 as fl

project = fl.Project.from_cloud("PROJECT_ID_HERE")

params: fl.SimulationParams = project.params

# Submit to the Virtual GPU queue with high scheduling priority
project.run_case(
    params=params,
    name="My vGPU case",
    billing_method="VirtualGPU",
    priority=8,
)
