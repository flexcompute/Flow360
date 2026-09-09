import flow360 as fl
import pandas as pd

case = fl.Case.from_cloud(case_id="your-case-id")

# the data structure holding the force coefficients
forces_by_surface = case.results.surface_forces

# exclude the wind tunnel elements
forces_by_surface.filter(exclude="*tunnel*")

# perform averaging
forces_total = forces_by_surface.get_averages(0.1)

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
force_coeffs = ['totalCL', 'totalCD', 'totalCFx', 'totalCFy', 'totalCFz',
                'totalCLPressure', 'totalCDPressure', 'totalCFxPressure', 'totalCFyPressure', 'totalCFzPressure',
                'totalCLSkinFriction', 'totalCDSkinFriction', 'totalCFxSkinFriction', 'totalCFySkinFriction', 'totalCFzSkinFriction']

for coeff in force_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * force_scale).to(fl.u.N)

# Convert moment coefficients to moments (N·m)
moment_x_coeffs = ['totalCMx', 'totalCMxPressure', 'totalCMxSkinFriction']

moment_y_coeffs = ['totalCMy', 'totalCMyPressure', 'totalCMySkinFriction']

moment_z_coeffs = ['totalCMz', 'totalCMzPressure', 'totalCMzSkinFriction']

for coeff in moment_x_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * moment_scale[0]).to(fl.u.N * fl.u.m)

for coeff in moment_y_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * moment_scale[1]).to(fl.u.N * fl.u.m)

for coeff in moment_z_coeffs:
    forces_actual[coeff.removeprefix("C")] = (forces_total[coeff] * moment_scale[2]).to(fl.u.N * fl.u.m)

# Rename the series to indicate these are actual forces/moments
forces_actual = pd.Series(forces_actual)

# Print the results
print("\n" + "="*60)
print("FORCE COEFFICIENTS - WIND TUNNEL EXCLUDED:")
print("="*60)
print(forces_total[forces_total.index.str.startswith("total")])

print("\n" + "="*60)
print("ACTUAL FORCES (N) AND MOMENTS (N·m) - WIND TUNNEL EXCLUDED:")
print("="*60)
print(forces_actual)
