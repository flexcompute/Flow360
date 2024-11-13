import flow360 as fl
from flow360 import u

fl.Env.preprod.active()

project = fl.Project.from_cloud("PROJECT_ID_HERE")

parent_case = fl.Case.from_cloud("PARENT_CASE_ID_HERE")

param: fl.SimulationParams = parent_case.params

# fork with new angle of attack being 1.23 degrees
param.operating_condition.alpha = 1.23 * u.deg

project.run_case(params=param, fork_from=parent_case, name="Forked case with new alpha")
