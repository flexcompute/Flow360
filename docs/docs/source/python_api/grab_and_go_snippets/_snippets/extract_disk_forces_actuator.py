import flow360 as fl

case = fl.Case.from_cloud(case_id="your-case-id")
results = case.results

# Non-dimensional actuator disk loads versus pseudo step.
print(results.actuator_disks.as_dataframe())

# Convert the loads to the SI unit system, then read them back as dimensional
# power, force and moment.
results.actuator_disks.to_base("SI")
actuator_disk_si = results.actuator_disks.as_dataframe()

actuator_disk_si.plot(
    x="pseudo_step",
    y=["Disk0_Power", "Disk0_Force", "Disk0_Moment"],
    xlabel="Pseudo Step",
    subplots=True,
    title="Actuator disk loads (SI)",
)
