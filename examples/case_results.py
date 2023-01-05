from pylab import *

from flow360.component.case import CaseList


my_cases = CaseList()
case = my_cases[0].to_case()

print(case)

# get residuals:
residuals = case.results.residuals

# print residuals as a table:
print(residuals.raw)

# print other results as tables:
print(case.results.cfl.raw)
print(case.results.total_forces.raw)
print(case.results.linear_residuals.raw)
print(case.results.minmax_state.raw)

# download all result files of the case:
case.results.download_manager(all=True)

# download specific result files of the case:
case.results.download_manager(bet_forces=True, actuator_disk_output=True)

# alternative way of downloading using dedicated functions:
case.results.download_surface()
case.results.download_volumetric()


try:
    case.results.total_forces.plot()
except NotImplementedError as e:
    print(e)

# plot some results manually, code will not re-download data, it is already stored in memory
plot(case.results.total_forces.raw["pseudo_step"], case.results.total_forces.raw["CL"])
xlabel("pseudo step")
ylabel("CL")
show()

# alterantive way of plotting using pre-defined plot setup
case.results.plot.total_forces()


try:
    case.results.total_forces.to_csv()
except NotImplementedError as e:
    print(e)
