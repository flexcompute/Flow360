import flow360 as fl

case = fl.Case.from_cloud(case_id="your-case-id")
results = case.results

# Integrated BET disk loads. to_base("SI") converts the non-dimensional solver
# output into dimensional forces and moments (e.g. Disk0_Force_x in Newtons).
results.bet_forces.to_base("SI")
print(results.bet_forces.as_dataframe())

# Radial distribution of the blade loading, resolved per disk and per blade.
# These are non-dimensional coefficients (thrust/torque coefficient per section).
bet_radial = results.bet_forces_radial_distribution.as_dataframe()
bet_radial.plot(
    x="Disk0_All_Radius",
    y=["Disk0_Blade0_All_ThrustCoeff", "Disk0_Blade0_All_TorqueCoeff"],
    xlabel="Radius",
    title="BET disk radial distribution",
)
