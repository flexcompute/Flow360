import pandas as pd

import flow360 as fl

case = fl.Case.from_cloud(case_id="your-case-id")

# the data structure holding the force coefficients, averaged over the last 10% of pseudo steps
forces_total = case.results.total_forces.get_averages(0.1)

# gather all reference values
density = case.params.operating_condition.thermal_state.density
reference_velocity = case.params.reference_velocity
reference_area = case.params.reference_geometry.area
reference_length = case.params.reference_geometry.moment_length

# Calculate dynamic pressure
dynamic_pressure = 0.5 * density * reference_velocity**2

# Calculate forces from force coefficients (F = Cf * q * A)
force_scale = dynamic_pressure * reference_area

# Calculate moments from moment coefficients (M = Cm * q * A * L)
moment_scale = force_scale * reference_length

# Create a new series with actual forces and moments
forces_actual = {}

# Convert force coefficients to forces (N)
force_coeffs = [
    "CL",
    "CD",
    "CFx",
    "CFy",
    "CFz",
    "CLPressure",
    "CDPressure",
    "CFxPressure",
    "CFyPressure",
    "CFzPressure",
    "CLSkinFriction",
    "CDSkinFriction",
    "CFxSkinFriction",
    "CFySkinFriction",
    "CFzSkinFriction",
]

for coeff in force_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * force_scale).to(fl.u.N)

# Convert moment coefficients to moments (N·m)
moment_x_coeffs = ["CMx", "CMxPressure", "CMxSkinFriction"]

moment_y_coeffs = ["CMy", "CMyPressure", "CMySkinFriction"]

moment_z_coeffs = ["CMz", "CMzPressure", "CMzSkinFriction"]

for coeff in moment_x_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * moment_scale[0]).to(
        fl.u.N * fl.u.m
    )

for coeff in moment_y_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * moment_scale[1]).to(
        fl.u.N * fl.u.m
    )

for coeff in moment_z_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * moment_scale[2]).to(
        fl.u.N * fl.u.m
    )

# Rename the series to indicate these are actual forces/moments
forces_actual = pd.Series(forces_actual)

# Print the results
print("\n" + "=" * 60)
print("FORCE COEFFICIENTS:")
print("=" * 60)
print(forces_total)

print("\n" + "=" * 60)
print("ACTUAL FORCES (N) AND MOMENTS (N·m):")
print("=" * 60)
print(forces_actual)
