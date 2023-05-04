import flow360 as fl

for case in fl.MyCases(limit=10000):
    print(case.id, case.status, case.name, case.info.userEmail)
    try:
        case.params
    except:
        case.params_as_dict
