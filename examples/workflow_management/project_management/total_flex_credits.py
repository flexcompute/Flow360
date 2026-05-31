import flow360 as fl

project_id = "prj-842fe363-fb66-4d33-85f2-384a88a448e7"  # Replace with your project ID

project = fl.Project.from_cloud(project_id=project_id)
case_ids = project.get_case_ids()
total_fc_cost = 0
for case in case_ids:
    case = fl.Case.from_cloud(case_id=case)
    print(f"{case.name}: cost = {case.info.computeCost}")
    total_fc_cost += case.info.computeCost

print(f"Total flex credits used for project {project_id}: {total_fc_cost}")
