import flow360 as fl

case = fl.Case.from_cloud(case_id="your-case-id")

# reference values to dimensionalize the coefficients (F = C * q * A)
density = case.params.operating_condition.thermal_state.density
reference_velocity = case.params.reference_velocity
reference_area = case.params.reference_geometry.area
force_scale = (0.5 * density * reference_velocity**2 * reference_area).to(fl.u.N)

# cumulative drag force (N) along X: the cumulative curve is an integrated
# coefficient, so C * q * A is a force.
x_dist = case.results.x_slicing_force_distribution
x_dist.wait()  # sectional post-processing can finish after the case
df_x = x_dist.as_dataframe()
df_x["Drag [N]"] = df_x["totalCumulative_CD_Curve"] * force_scale.value
df_x.plot(x="X", y="Drag [N]", title="Cumulative drag along X")

# spanwise lift loading (N/m) along Y: CFz_per_span is a per-span coefficient,
# so C * q * A is a force per unit span, not a total force.
y_dist = case.results.y_slicing_force_distribution
y_dist.wait()
df_y = y_dist.as_dataframe()
df_y["Lift [N/m]"] = df_y["totalCFz_per_span"] * force_scale.value
df_y.plot(x="Y", y="Lift [N/m]", title="Spanwise lift loading along Y")
