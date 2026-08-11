import flow360 as fl

# Get case from cloud
case = fl.Case.from_cloud(case_id="your-case-id")

# Get surface forces and filter to a specific boundary
surface_forces = case.results.surface_forces
surface_forces.filter(include="wing")  # supports wildcards, e.g., "wing*"

# Average over the last 10% of iterations
averaged_forces = surface_forces.get_averages(0.1)

# Print lift and drag coefficients
print(f"CL: {averaged_forces['totalCL']}")
print(f"CD: {averaged_forces['totalCD']}")
