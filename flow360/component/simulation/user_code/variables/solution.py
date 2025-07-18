"""Solution variables of Flow360"""

import unyt as u

from flow360.component.simulation.user_code.core.types import SolverVariable

# pylint:disable = fixme
# TODO:Scalar type (needs further discussion on how to handle scalar values)
# bet_thrust = SolverVariable(
#     name="solution.bet_thrust", value=float("NaN")
# )  # Thrust force for BET disk
# bet_torque = SolverVariable(name="solution.bet_torque", value=float("NaN"))  # Torque for BET disk
# bet_omega = SolverVariable(
#     name="solution.bet_omega", value=float("NaN")
# )  # Rotation speed for BET disk
# CD = SolverVariable(name="solution.CD", value=float("NaN"))  # Drag coefficient on patch
# CL = SolverVariable(name="solution.CL", value=float("NaN"))  # Lift coefficient on patch
# forceX = SolverVariable(name="solution.forceX", value=float("NaN"))  # Total force in X direction
# forceY = SolverVariable(name="solution.forceY", value=float("NaN"))  # Total force in Y direction
# forceZ = SolverVariable(name="solution.forceZ", value=float("NaN"))  # Total force in Z direction
# momentX = SolverVariable(name="solution.momentX", value=float("NaN"))  # Total moment in X direction
# momentY = SolverVariable(name="solution.momentY", value=float("NaN"))  # Total moment in Y direction
# momentZ = SolverVariable(name="solution.momentZ", value=float("NaN"))  # Total moment in Z direction


# pylint:disable=no-member
# Common
coordinate = SolverVariable(
    name="solution.coordinate",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m,
    solver_name="coordinate",
    variable_type="Volume",
)  # Grid coordinates

Cp = SolverVariable(
    name="solution.Cp",
    value=float("NaN"),
    solver_name="___Cp",
    variable_type="Volume",
)
Cpt = SolverVariable(
    name="solution.Cpt",
    value=float("NaN"),
    solver_name="___Cpt",
    variable_type="Volume",
)
grad_density = SolverVariable(
    name="solution.grad_density",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.kg / u.m**4,
    solver_name="___grad_density",
    variable_type="Volume",
)
grad_u = SolverVariable(
    name="solution.grad_u",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="___grad_u",
    variable_type="Volume",
)
grad_v = SolverVariable(
    name="solution.grad_v",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="___grad_v",
    variable_type="Volume",
)
grad_w = SolverVariable(
    name="solution.grad_w",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="___grad_w",
    variable_type="Volume",
)
grad_pressure = SolverVariable(
    name="solution.grad_pressure",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.Pa / u.m,
    solver_name="___grad_pressure",
    variable_type="Volume",
)

Mach = SolverVariable(
    name="solution.Mach",
    value=float("NaN"),
    solver_name="___Mach",
    variable_type="Volume",
)
mut = SolverVariable(
    name="solution.mut",
    value=float("NaN") * u.kg / u.m / u.s,
    solver_name="___mut",
    variable_type="Volume",
)  # Turbulent viscosity
mut_ratio = SolverVariable(
    name="solution.mut_ratio",
    value=float("NaN"),
    solver_name="___mut_ratio",
    variable_type="Volume",
)
nu_hat = SolverVariable(
    name="solution.nu_hat",
    value=float("NaN") * u.m**2 / u.s,
    solver_name="___nu_hat",
    variable_type="Volume",
)
turbulence_kinetic_energy = SolverVariable(
    name="solution.turbulence_kinetic_energy",
    value=float("NaN") * u.J / u.kg,
    solver_name="___turbulence_kinetic_energy",
    variable_type="Volume",
)  # k
specific_rate_of_dissipation = SolverVariable(
    name="solution.specific_rate_of_dissipation",
    value=float("NaN") / u.s,
    solver_name="___specific_rate_of_dissipation",
    variable_type="Volume",
)  # Omega
amplification_factor = SolverVariable(
    name="solution.amplification_factor",
    value=float("NaN"),
    solver_name="___amplification_factor",
    variable_type="Volume",
)  # transition model variable: n, non-dimensional
turbulence_intermittency = SolverVariable(
    name="solution.turbulence_intermittency",
    value=float("NaN"),
    solver_name="___turbulence_intermittency",
    variable_type="Volume",
)  # transition model variable: gamma, non-dimensional


density = SolverVariable(
    name="solution.density",
    value=float("NaN") * u.kg / u.m**3,
    solver_name="___density",
    variable_type="Volume",
)
velocity = SolverVariable(
    name="solution.velocity",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m / u.s,
    solver_name="___velocity",
    variable_type="Volume",
)
pressure = SolverVariable(
    name="solution.pressure",
    value=float("NaN") * u.Pa,
    solver_name="___pressure",
    variable_type="Volume",
)

qcriterion = SolverVariable(
    name="solution.qcriterion",
    value=float("NaN") / u.s**2,
    solver_name="___qcriterion",
    variable_type="Volume",
)
entropy = SolverVariable(
    name="solution.entropy",
    value=float("NaN") * u.J / u.K,
    solver_name="___entropy",
    variable_type="Volume",
)
temperature = SolverVariable(
    name="solution.temperature",
    value=float("NaN") * u.K,
    solver_name="___temperature",
    variable_type="Volume",
)
vorticity = SolverVariable(
    name="solution.vorticity",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="___vorticity",
    variable_type="Volume",
)
wall_distance = SolverVariable(
    name="solution.wall_distance",
    value=float("NaN") * u.m,
    solver_name="wallDistance",
    variable_type="Volume",
)

# Surface
CfVec = SolverVariable(
    name="solution.CfVec",
    value=[float("NaN"), float("NaN"), float("NaN")],
    solver_name="___CfVec",
    variable_type="Surface",
)
Cf = SolverVariable(
    name="solution.Cf",
    value=float("NaN"),
    solver_name="___Cf",
    variable_type="Surface",
)
heat_flux = SolverVariable(
    name="solution.heat_flux",
    value=float("NaN") * u.W / u.m**2,
    solver_name="heatFlux",
    variable_type="Surface",
)
node_area_vector = SolverVariable(
    name="solution.node_area_vector",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m**2,
    solver_name="nodeNormals",
    variable_type="Surface",
)
node_unit_normal = SolverVariable(
    name="solution.node_unit_normal",
    value=[float("NaN"), float("NaN"), float("NaN")],
    solver_name="___node_unit_normal",
    variable_type="Surface",
)
node_forces_per_unit_area = SolverVariable(
    name="solution.node_forces_per_unit_area",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.Pa,
    solver_name="___node_forces_per_unit_area",
    variable_type="Surface",
)
y_plus = SolverVariable(
    name="solution.y_plus", value=float("NaN"), solver_name="yPlus", variable_type="Surface"
)
wall_shear_stress_magnitude = SolverVariable(
    name="solution.wall_shear_stress_magnitude",
    value=float("NaN") * u.Pa,
    solver_name="___wall_shear_stress_magnitude",
    variable_type="Surface",
)
heat_transfer_coefficient_static_temperature = SolverVariable(
    name="solution.heat_transfer_coefficient_static_temperature",
    value=float("NaN") * u.W / (u.m**2 * u.K),
    solver_name="___heat_transfer_coefficient_static_temperature",
    variable_type="Surface",
)
heat_transfer_coefficient_total_temperature = SolverVariable(
    name="solution.heat_transfer_coefficient_total_temperature",
    value=float("NaN") * u.W / (u.m**2 * u.K),
    solver_name="___heat_transfer_coefficient_total_temperature",
    variable_type="Surface",
)


# TODO
# pylint:disable = fixme
# velocity_relative = SolverVariable(
#     name="solution.velocity_relative",
#     value=[float("NaN"), float("NaN"), float("NaN")] * u.m / u.s,
#     solver_name="velocityRelative",
#     prepending_code="double velocityRelative[3];for(int i=0;i<3;i++)"
#     + "{velocityRelative[i]=velocity[i]-nodeVelocity[i];}",
#     variable_type="Volume",
# )
# wallFunctionMetric = SolverVariable(
#     name="solution.wallFunctionMetric", value=float("NaN"), variable_type="Surface"
# )
# bet_metrics_alpha_degree = SolverVariable(
#     name="solution.bet_metrics_alpha_degree", value=float("NaN") * u.deg, variable_type="Volume"
# )
# bet_metrics_Cf_axial = SolverVariable(
#     name="solution.bet_metrics_Cf_axial", value=float("NaN"), variable_type="Volume"
# )
# bet_metrics_Cf_circumferential = SolverVariable(
#     name="solution.bet_metrics_Cf_circumferential", value=float("NaN"), variable_type="Volume"
# )
# bet_metrics_local_solidity_integral_weight = SolverVariable(
#     name="solution.bet_metrics_local_solidity_integral_weight",
#     value=float("NaN"),
#     variable_type="Volume",
# )
# bet_metrics_tip_loss_factor = SolverVariable(
#     name="solution.bet_metrics_tip_loss_factor", value=float("NaN"), variable_type="Volume"
# )
# bet_metrics_velocity_relative = SolverVariable(
#     name="solution.bet_metrics_velocity_relative",
#     value=[float("NaN"), float("NaN"), float("NaN")] * u.m / u.s,
#     variable_type="Volume",
# )
# betMetricsPerDisk = SolverVariable(
#     name="solution.betMetricsPerDisk", value=float("NaN"), variable_type="Volume"
# )


# Abandoned (Possible)
# SpalartAllmaras_hybridModel = SolverVariable(
#     name="solution.SpalartAllmaras_hybridModel", value=float("NaN"), variable_type="Volume"
# )
# kOmegaSST_hybridModel = SolverVariable(
#     name="solution.kOmegaSST_hybridModel", value=float("NaN"), variable_type="Volume"
# )
# localCFL = SolverVariable(name="solution.localCFL", value=float("NaN"), variable_type="Volume")
# numericalDissipationFactor = SolverVariable(
#     name="solution.numericalDissipationFactor", value=float("NaN"), variable_type="Volume"
# )
# lowMachPreconditionerSensor = SolverVariable(
#     name="solution.lowMachPreconditionerSensor", value=float("NaN"), variable_type="Volume"
# )

# Abandoned
# linearResidualNavierStokes = SolverVariable(
#     name="solution.linearResidualNavierStokes", value=float("NaN"), variable_type="Volume"
# )
# linearResidualTurbulence = SolverVariable(
#     name="solution.linearResidualTurbulence", value=float("NaN"), variable_type="Volume"
# )
# linearResidualTransition = SolverVariable(
#     name="solution.linearResidualTransition", value=float("NaN"), variable_type="Volume"
# )
# residualNavierStokes = SolverVariable(
#     name="solution.residualNavierStokes", value=float("NaN"), variable_type="Volume"
# )
# residualTransition = SolverVariable(
#     name="solution.residualTransition", value=float("NaN"), variable_type="Volume"
# )
# residualTurbulence = SolverVariable(
#     name="solution.residualTurbulence", value=float("NaN"), variable_type="Volume"
# )
# solutionNavierStokes = SolverVariable(
#     name="solution.solutionNavierStokes", value=float("NaN"), variable_type="Volume"
# )
# solutionTurbulence = SolverVariable(
#     name="solution.solutionTurbulence", value=float("NaN"), variable_type="Volume"
# )
# residualHeatSolver = SolverVariable(
#     name="solution.residualHeatSolver", value=float("NaN"), variable_type="Volume"
# )
# velocity_x = SolverVariable(name="solution.velocity_x", value=float("NaN"), variable_type="Volume")
# velocity_y = SolverVariable(name="solution.velocity_y", value=float("NaN"), variable_type="Volume")
# velocity_z = SolverVariable(name="solution.velocity_z", value=float("NaN"), variable_type="Volume")
# velocity_magnitude = SolverVariable(
#     name="solution.velocity_magnitude", value=float("NaN"), variable_type="Volume"
# )
# vorticityMagnitude = SolverVariable(
#     name="solution.vorticityMagnitude", value=float("NaN"), variable_type="Volume"
# )
# vorticity_x = SolverVariable(
#     name="solution.vorticity_x", value=float("NaN"), variable_type="Volume"
# )
# vorticity_y = SolverVariable(
#     name="solution.vorticity_y", value=float("NaN"), variable_type="Volume"
# )
# vorticity_z = SolverVariable(
#     name="solution.vorticity_z", value=float("NaN"), variable_type="Volume"
# )
# wall_shear_stress_magnitude_pa = SolverVariable(
#     name="solution.wall_shear_stress_magnitude_pa", value=float("NaN"), variable_type="Surface"
# )
