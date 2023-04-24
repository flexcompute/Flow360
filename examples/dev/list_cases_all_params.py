import flow360 as fl

for case in fl.MyCases(limit=10000):
    print(case.id, case.status, case.name, case.info.userEmail)
    case.params

