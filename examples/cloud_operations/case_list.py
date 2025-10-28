import flow360.v1 as fl
from flow360.log import log

for case in fl.MyCases(limit=1000):
    log.info(case.short_description() + "solver_version = " + str(case.solver_version))
