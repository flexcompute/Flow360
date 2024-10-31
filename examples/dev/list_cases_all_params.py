import flow360.component.v1 as fl
from flow360.exceptions import Flow360ValidationError

for case in fl.MyCases(limit=10000):
    print(case.id, case.status, case.name, case.info.userEmail)
    try:
        case.params
    except Flow360ValidationError:
        case.params_as_dict
