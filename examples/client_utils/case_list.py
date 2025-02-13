import flow360.v1 as fl
from flow360.log import log

for case in fl.MyCases(limit=10000):
    log.info(
        "\n"
        + "Case ID: "
        + str(case.id)
        + "\n"
        + "Case name: "
        + str(case.name)
        + "\n"
        + "Status: "
        + str(case.status.value)
        + "\n"
        + "Solver version: "
        + str(case.solver_version)
    )
