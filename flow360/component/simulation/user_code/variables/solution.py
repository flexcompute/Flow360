"""Solution variables of Flow360"""

import numpy as np
import unyt as u

from flow360.component.simulation.user_code.core.types import SolverVariable

# pylint:disable = fixme
# TODO:Scalar type (needs further discussion)
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


# Common
coordinate = SolverVariable(
    name="solution.coordinate",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m,
    solver_name="nodePosition",
    variable_type="Volume",
)  # Grid coordinates

Cp = SolverVariable(
    name="solution.Cp",
    value=float("NaN"),
    solver_name="Cp",
    prepending_code="double Cp;Cp=(primitive[4]-pressureFreestream)/(0.5*MachRef*MachRef);",
    variable_type="Volume",
)
Cpt = SolverVariable(
    name="solution.Cpt",
    value=float("NaN"),
    solver_name="Cpt",
    prepending_code="double Cpt;double MachUser=sqrt(primitive[1]*primitive[1]+"
    + "primitive[2]*primitive[2]+primitive[3]*primitive[3])"
    + "/sqrt(gamma*primitive[4]/primitive[0]);"
    + "Cpt=(gamma*primitive[4]*pow(1.0+(gamma-1.0)/2.*MachUser*MachUser,"
    + "gamma/(gamma-1.))-pow(1.0+(gamma-1.0)/2.*MachRef*MachRef,"
    + "gamma/(gamma-1.)))/(0.5*gamma*MachRef*MachRef);",
    variable_type="Volume",
)
grad_density = SolverVariable(
    name="solution.grad_density",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.kg / u.m**4,
    solver_name="gradDensity",
    prepending_code="double gradDensity[3];gradDensity[0]=gradPrimitive[0][0];"
    + "gradDensity[1]=gradPrimitive[0][1];gradDensity[2]=gradPrimitive[0][2];",
    variable_type="Volume",
)
grad_velocity_x = SolverVariable(
    name="solution.grad_velocity_x",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="gradVelocityX",
    prepending_code="double gradVelocityX[3];gradVelocityX[0]=gradPrimitive[1][0];"
    + "gradVelocityX[1]=gradPrimitive[1][1];gradVelocityX[2]=gradPrimitive[1][2];",
    variable_type="Volume",
)
grad_velocity_y = SolverVariable(
    name="solution.grad_velocity_y",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="gradVelocityY",
    prepending_code="double gradVelocityY[3];gradVelocityY[0]=gradPrimitive[2][0];"
    + "gradVelocityY[1]=gradPrimitive[2][1];gradVelocityY[2]=gradPrimitive[2][2];",
    variable_type="Volume",
)
gradVelocity_z = SolverVariable(
    name="solution.grad_velocity_z",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="gradVelocityZ",
    prepending_code="double gradVelocityZ[3];gradVelocityZ[0]=gradPrimitive[3][0];"
    + "gradVelocityZ[1]=gradPrimitive[3][1];gradVelocityZ[2]=gradPrimitive[3][2];",
    variable_type="Volume",
)
grad_pressure = SolverVariable(
    name="solution.grad_pressure",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.Pa / u.m,
    solver_name="gradPressure",
    prepending_code="double gradPressure[3];gradPressure[0]=gradPrimitive[4][0];"
    + "gradPressure[1]=gradPrimitive[4][1];gradPressure[2]=gradPrimitive[4][2];",
    variable_type="Volume",
)

Mach = SolverVariable(
    name="solution.Mach",
    value=float("NaN"),
    solver_name="Mach",
    prepending_code="double Mach;Mach=sqrt(primitive[1]*primitive[1]+"
    + "primitive[2]*primitive[2]+primitive[3]*primitive[3])"
    + "/sqrt(gamma*primitive[4]/primitive[0]);"
    + "if (usingLiquidAsMaterial){Mach=0;}",
    variable_type="Volume",
)
mut = SolverVariable(
    name="solution.mut",
    value=float("NaN") * u.kg / u.m / u.s,
    solver_name="mut",
    variable_type="Volume",
)  # Turbulent viscosity
mu = SolverVariable(
    name="solution.mu",
    value=float("NaN") * u.kg / u.m / u.s,
    solver_name="mu",
    variable_type="Volume",
)  # Laminar viscosity
mut_ratio = SolverVariable(
    name="solution.mut_ratio",
    value=float("NaN"),
    solver_name="mutRatio",
    prepending_code="double mutRatio;mutRatio=mut/mu",
    variable_type="Volume",
)
nu_hat = SolverVariable(
    name="solution.nu_hat",
    value=float("NaN") * u.m**2 / u.s,
    solver_name="SpalartAllmaras_solution",
    variable_type="Volume",
)
turbulence_kinetic_energy = SolverVariable(
    name="solution.turbulence_kinetic_energy",
    value=float("NaN") * u.J / u.kg,
    solver_name="kOmegaSST_solution[0]",
    variable_type="Volume",
)  # k
specific_rate_of_dissipation = SolverVariable(
    name="solution.specific_rate_of_dissipation",
    value=float("NaN") / u.s,
    solver_name="kOmegaSST_solution[1]",
    variable_type="Volume",
)  # Omega
amplification_factor = SolverVariable(
    name="solution.amplification_factor",
    value=float("NaN"),
    solver_name="solutionTransition[0]",
    variable_type="Volume",
)  # transition model variable: n, non-dimensional
turbulence_intermittency = SolverVariable(
    name="solution.turbulence_intermittency",
    value=float("NaN"),
    solver_name="solutionTransition[1]",
    variable_type="Volume",
)  # transition model variable: gamma, non-dimensional


density = SolverVariable(
    name="solution.density",
    value=float("NaN") * u.kg / u.m**3,
    solver_name="primitive[0]",
    variable_type="Volume",
)
velocity = SolverVariable(
    name="solution.velocity",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m / u.s,
    solver_name="velocity",
    prepending_code="double velocity[3];"
    + "velocity[0]=primitive[1]*velocityScale;"
    + "velocity[1]=primitive[2]*velocityScale;"
    + "velocity[2]=primitive[3]*velocityScale;",
    variable_type="Volume",
)
pressure = SolverVariable(
    name="solution.pressure",
    value=float("NaN") * u.Pa,
    solver_name="primitive[4]",
    variable_type="Volume",
)

qcriterion = SolverVariable(
    name="solution.qcriterion",
    value=float("NaN") / u.s**2,
    solver_name="qcriterion",
    prepending_code="double qcriterion;"
    + "double ux=gradPrimitive[1][0];"
    + "double uy=gradPrimitive[1][1];"
    + "double uz=gradPrimitive[1][2];"
    + "double vx=gradPrimitive[2][0];"
    + "double vy=gradPrimitive[2][1];"
    + "double vz=gradPrimitive[2][2];"
    + "double wx=gradPrimitive[3][0];"
    + "double wy=gradPrimitive[3][1];"
    + "double wz=gradPrimitive[3][2];"
    + "double str11=ux;"
    + "double str22=vy;"
    + "double str33=wz;"
    + "double str12=0.5*(uy+vx);"
    + "double str13=0.5*(uz+wx);"
    + "double str23=0.5*(vz+wy);"
    + "double str_norm=str11*str11+str22*str22+str33*str33+2*(str12*str12)+2*(str13*str13)+2*(str23*str23);"
    + "double omg12=0.5*(uy-vx);"
    + "double omg13=0.5*(uz-wx);"
    + "double omg23=0.5*(vz-wy);"
    + "double omg_norm=2*(omg12*omg12)+2*(omg13*omg13)+2*(omg23*omg23);"
    + "qcriterion=0.5*(omg_norm-str_norm)*(velocityScale*velocityScale);",
    variable_type="Volume",
)
entropy = SolverVariable(
    name="solution.entropy",
    value=float("NaN") * u.J / u.K,
    prepending_code="double entropy;entropy=log(primitive[4]/gasConstant/pow(primitive[0],gamma))",
    solver_name="entropy",
    variable_type="Volume",
)
temperature = SolverVariable(
    name="solution.temperature",
    value=float("NaN") * u.K,
    prepending_code="double temperature;temperature=primitive[4]/(primitive[0]* gasConstant);",
    solver_name="temperature",
    variable_type="Volume",
)
vorticity = SolverVariable(
    name="solution.vorticity",
    value=[float("NaN"), float("NaN"), float("NaN")] / u.s,
    solver_name="vorticity",
    prepending_code="double vorticity[3];"
    + "vorticity[0]=(gradPrimitive[3][1] - gradPrimitive[2][2]) * velocityScale;"
    + "vorticity[1]=(gradPrimitive[1][2] - gradPrimitive[3][0]) * velocityScale;"
    + "vorticity[2]=(gradPrimitive[2][0] - gradPrimitive[1][1]) * velocityScale;",
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
    solver_name="CfVec",
    prepending_code="double CfVec[3];"
    + "for(int i=0;i<3;i++)"
    + "{CfVec[i]=wallShearStress[i]/(0.5*MachRef*MachRef);}",
    variable_type="Surface",
)
Cf = SolverVariable(
    name="solution.Cf",
    value=float("NaN"),
    solver_name="Cf",
    prepending_code="double Cf;Cf=magnitude(wallShearStress)/(0.5*MachRef*MachRef);",
    variable_type="Surface",
)
heatflux = SolverVariable(
    name="solution.heatflux",
    value=float("NaN") * u.W / u.m**2,
    solver_name="heatFlux",
    variable_type="Surface",
)
node_normals = SolverVariable(
    name="solution.node_normals",
    value=[float("NaN"), float("NaN"), float("NaN")],
    solver_name="nodeNormals",
    variable_type="Surface",
)
node_forces_per_unit_area = SolverVariable(
    name="solution.node_forces_per_unit_area",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.Pa,
    solver_name="nodeForcesPerUnitArea",
    prepending_code="double nodeForcesPerUnitArea[3];"
    + "double normalMag=magnitude(nodeNormals);"
    + "for(int i=0;i<3;i++){nodeForcesPerUnitArea[i]="
    + "((primitive[4]-pressureFreestream)*nodeNormals[i]/normalMag+wallViscousStress[i])"
    + "*(velocityScale*velocityScale);}",
    variable_type="Surface",
)
y_plus = SolverVariable(
    name="solution.y_plus", value=float("NaN"), solver_name="yPlus", variable_type="Surface"
)
wall_shear_stress = SolverVariable(
    name="solution.wall_shear_stress",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.Pa,
    solver_name="wallShearStress",
    variable_type="Surface",
)
heat_transfer_coefficient_static_temperature = SolverVariable(
    name="solution.heat_transfer_coefficient_static_temperature",
    value=float("NaN") * u.W / (u.m**2 * u.K),
    solver_name="heatTransferCoefficientStaticTemperature",
    prepending_code="double heatTransferCoefficientStaticTemperature;"
    + "double temperature=primitive[4]/(primitive[0]*gasConstant);"
    + f"double temperatureSafeDivide; double epsilon={np.finfo(np.float64).eps};"
    + "heatTransferCoefficientTotalTemperature={1.0/epsilon};"
    + "if(temperature-1.0<0){temperatureSafeDivide=temperature-1.0-epsilon;}"
    + "else{temperatureSafeDivide=temperature-1.0+epsilon;}"
    + "if(abs(temperature-temperatureTotal)>epsilon)"
    + "{temperatureTotal=-heatFlux/temperatureSafeDivide;}",
    variable_type="Surface",
)
heat_transfer_coefficient_total_temperature = SolverVariable(
    name="solution.heat_transfer_coefficient_total_temperature",
    value=float("NaN") * u.W / (u.m**2 * u.K),
    solver_name="heatTransferCoefficientTotalTemperature",
    prepending_code="double heatTransferCoefficientTotalTemperature;"
    + "double temperature=primitive[4]/(primitive[0]*gasConstant);"
    + "double temperatureTotal = 1.0 + (gamma - 1.0) / 2.0 * MachRef * MachRef;"
    + f"double temperatureSafeDivide; double epsilon={np.finfo(np.float64).eps};"
    + "if(temperature-temperatureTotal<0){temperatureSafeDivide=temperature-temperatureTotal-epsilon;}"
    + "else{temperatureSafeDivide=temperature-temperatureTotal+epsilon;}"
    + "heatTransferCoefficientTotalTemperature=1.0/epsilon;"
    + "if(abs(temperature-temperatureTotal)>epsilon)"
    + "{temperatureTotal=-heatFlux/temperatureSafeDivide;}",
    variable_type="Surface",
)


# TODO
# pylint:disable = fixme
# velocity_relative = SolverVariable(
#     name="solution.velocity_relative",
#     value=[float("NaN"), float("NaN"), float("NaN")] * u.m / u.s,
#     solver_name="velocityRelative",
#     prepending_code="double velocityRelative[3];for(int i=0;i<3;i++){velocityRelative[i]=velocity[i]-nodeVelocity[i];}",
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
