import flow360 as fl

# Get case from cloud
case = fl.Case.from_cloud(case_id="case-8f163865-b907-4e16-a082-e046bb4a5f99")

# List available monitors
print(case.results.monitors.monitor_names)

# Names from output configuration
probe_output_name = "probe_output_1"
probe_name = "probe_1"
variable_name = "pressure_in_SI"

# Get a specific probe monitor by name (use the name of the probe output)
probe = case.results.monitors["probe_output_1"]

# Include time column (for unsteady simulations)
probe.reload_data(include_time=True, filter_physical_steps_only=True)

# Get as pandas DataFrame
df = probe.as_dataframe()

# Plot pressure vs time
df.plot(
    x="time",
    y=f"{probe_output_name}_{probe_name}_{variable_name}",
    title="Pressure Time History",
    xlabel="Time [s]",
    ylabel="Pressure",
)
