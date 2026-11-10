import flow360.v1 as fl

for case in fl.MyCases(limit=1000):
    print(case.short_description() + "solver_version = " + str(case.solver_version))
