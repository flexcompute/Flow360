import flow360 as fl

case=fl.Case.from_cloud("case-69868332-6f72-4c21-bb55-c885d27c4960")
param=case.params
param.private_attribute_dict={"liquidOutputOriginalDensity":True}

prj=fl.Project.from_cloud(case.project_id)

prj.run_case(params=param, name="ma003test", solver_version="Mach0003-25.9.99", use_beta_mesher=True )
