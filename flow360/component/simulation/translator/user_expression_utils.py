"""Utilities for user expression translation."""

import numpy as np

udf_prepending_code = {
    "solution.Cp": "double ___Cp = (primitiveVars[4] - pressureFreestream) / (0.5 * MachRef * MachRef);",
    "solution.Cpt": "double ___MachTmp = sqrt(primitiveVars[1] * primitiveVars[1] + "
    + "primitiveVars[2] * primitiveVars[2] + primitiveVars[3] * primitiveVars[3]) / "
    + "sqrt(1.4 * primitiveVars[4] / primitiveVars[0]);"
    + "double ___Cpt = (1.4 * primitiveVars[4] * pow(1.0 + (1.4 - 1.0) / 2. * ___MachTmp * ___MachTmp,"
    + "1.4 / (1.4 - 1.0)) - pow(1.0 + (1.4 - 1.0) / 2. * MachRef * MachRef,"
    + "1.4 / (1.4 - 1.0))) / (0.5 * 1.4 * MachRef * MachRef);",
    "solution.grad_density": "double ___grad_density[3]; ___grad_density[0] = gradPrimitive[0][0];"
    + "___grad_density[1] = gradPrimitive[0][1];"
    + "___grad_density[2] = gradPrimitive[0][2];",
    "solution.grad_u": "double ___grad_u[3];"
    + "___grad_u[0] = gradPrimitive[1][0] * velocityScale;"
    + "___grad_u[1] = gradPrimitive[1][1] * velocityScale;"
    + "___grad_u[2] = gradPrimitive[1][2] * velocityScale;",
    "solution.grad_v": "double ___grad_v[3];"
    + "___grad_v[0] = gradPrimitive[2][0] * velocityScale;"
    + "___grad_v[1] = gradPrimitive[2][1] * velocityScale;"
    + "___grad_v[2] = gradPrimitive[2][2] * velocityScale;",
    "solution.grad_w": "double ___grad_w[3];"
    + "___grad_w[0] = gradPrimitive[3][0] * velocityScale;"
    + "___grad_w[1] = gradPrimitive[3][1] * velocityScale;"
    + "___grad_w[2] = gradPrimitive[3][2] * velocityScale;",
    "solution.grad_pressure": "double ___grad_pressure[3];"
    + "___grad_pressure[0] = gradPrimitive[4][0];"
    + "___grad_pressure[1] = gradPrimitive[4][1];"
    + "___grad_pressure[2] = gradPrimitive[4][2];",
    "solution.Mach": "double ___Mach;"
    + "___Mach = usingLiquidAsMaterial ? 0 : "
    + "sqrt(primitiveVars[1] * primitiveVars[1] + "
    + "primitiveVars[2] * primitiveVars[2] + "
    + "primitiveVars[3] * primitiveVars[3]) / "
    + "sqrt(1.4 * primitiveVars[4] / primitiveVars[0]);",
    "solution.mut": "double ___mut; ___mut = mut * velocityScale;",
    "solution.mut_ratio": "double ___mut_ratio; ___mut_ratio = mut / mu;",
    "solution.nu_hat": "double ___nu_hat;"
    + "___nu_hat = solutionTurbulence * velocityScale * SpalartAllmaras_solutionRescale_0;",
    "solution.turbulence_kinetic_energy": "double ___turbulence_kinetic_energy;"
    "___turbulence_kinetic_energy = solutionTurbulence[0] * pow(velocityScale, 2) * kOmegaSST_solutionRescale_0;",
    "solution.specific_rate_of_dissipation": "double ___specific_rate_of_dissipation;"
    + "___specific_rate_of_dissipation = solutionTurbulence[1] * velocityScale * kOmegaSST_solutionRescale_1;",
    "solution.amplification_factor": "double ___amplification_factor;"
    "___amplification_factor = solutionTransition[0] * AmplificationFactorTransport_solutionRescale_0;",
    "solution.turbulence_intermittency": "double ___turbulence_intermittency;"
    + "___turbulence_intermittency = solutionTransition[1] * AmplificationFactorTransport_solutionRescale_1;",
    "solution.density": "double ___density;"
    + "___density = usingLiquidAsMaterial ? 1.0 : primitiveVars[0];",
    "solution.velocity": "double ___velocity[3];"
    + "___velocity[0] = primitiveVars[1] * velocityScale;"
    + "___velocity[1] = primitiveVars[2] * velocityScale;"
    + "___velocity[2] = primitiveVars[3] * velocityScale;",
    "solution.pressure": "double ___pressure;"
    + "___pressure = usingLiquidAsMaterial ? (primitiveVars[4] - 1.4 / 1.0) * "
    "(velocityScale * velocityScale) : primitiveVars[4];",
    "solution.qcriterion": "double ___qcriterion;"
    + "double ___ux = gradPrimitive[1][0];"
    + "double ___uy = gradPrimitive[1][1];"
    + "double ___uz = gradPrimitive[1][2];"
    + "double ___vx = gradPrimitive[2][0];"
    + "double ___vy = gradPrimitive[2][1];"
    + "double ___vz = gradPrimitive[2][2];"
    + "double ___wx = gradPrimitive[3][0];"
    + "double ___wy = gradPrimitive[3][1];"
    + "double ___wz = gradPrimitive[3][2];"
    + "double ___str11 = ___ux;"
    + "double ___str22 = ___vy;"
    + "double ___str33 = ___wz;"
    + "double ___str12 = 0.5 * (___uy + ___vx);"
    + "double ___str13 = 0.5 * (___uz + ___wx);"
    + "double ___str23 = 0.5 * (___vz + ___wy);"
    + "double ___str_norm = ___str11 * ___str11 + ___str22 * ___str22 + ___str33 * ___str33 + "
    + "2 * (___str12 * ___str12) + 2 * (___str13 * ___str13) + 2 * (___str23 * ___str23);"
    + "double ___omg12 = 0.5 * (___uy - ___vx);"
    + "double ___omg13 = 0.5 * (___uz - ___wx);"
    + "double ___omg23 = 0.5 * (___vz - ___wy);"
    + "double ___omg_norm = 2 * (___omg12 * ___omg12) + 2 * (___omg13 * ___omg13) + 2 * (___omg23 * ___omg23);"
    + "___qcriterion = 0.5 * (___omg_norm - ___str_norm) * (velocityScale * velocityScale);",
    "solution.entropy": "double ___entropy;"
    + "___entropy = log(primitiveVars[4] / (1.0 / 1.4) / pow(primitiveVars[0], 1.4));",
    "solution.temperature": "double ___temperature;"
    "___temperature =  primitiveVars[4] / (primitiveVars[0] * (1.0 / 1.4));",
    "solution.temperature_solid": "double ___temperature;"
    + f"double ___epsilon = {np.finfo(np.float64).eps};"
    "___temperature = (primitiveVars[0] < ___epsilon) ? "
    "solutionHeatSolver : primitiveVars[4] / (primitiveVars[0] * (1.0 / 1.4));",
    "solution.vorticity": "double ___vorticity[3];"
    + "___vorticity[0] = (gradPrimitive[3][1] - gradPrimitive[2][2]) * velocityScale;"
    + "___vorticity[1] = (gradPrimitive[1][2] - gradPrimitive[3][0]) * velocityScale;"
    + "___vorticity[2] = (gradPrimitive[2][0] - gradPrimitive[1][1]) * velocityScale;",
    "solution.CfVec": "double ___CfVec[3]; for (int i = 0; i < 3; i++)"
    + "{___CfVec[i] = wallShearStress[i] / (0.5 * MachRef * MachRef);}",
    "solution.Cf": "double ___Cf;"
    + "___Cf = magnitude(wallShearStress) / (0.5 * MachRef * MachRef);",
    "solution.node_unit_normal": "double ___node_unit_normal[3];"
    + "double ___normalMag = magnitude(nodeNormals);"
    + "for (int i = 0; i < 3; i++){___node_unit_normal[i] = "
    + "nodeNormals[i] / ___normalMag;}",
    "solution.node_forces_per_unit_area": "double ___node_forces_per_unit_area[3];"
    + "double ___normalMag = magnitude(nodeNormals);"
    + "for (int i = 0; i < 3; i++){___node_forces_per_unit_area[i] = "
    + "((primitiveVars[4] - pressureFreestream) * nodeNormals[i] / ___normalMag + wallViscousStress[i])"
    + " * (velocityScale * velocityScale);}",
    "solution.heat_transfer_coefficient_static_temperature": "double ___heat_transfer_coefficient_static_temperature;"
    + "double ___temperatureTmp = "
    + "primitiveVars[4] / (primitiveVars[0] * 1.0 / 1.4);"
    + f"double ___epsilon = {np.finfo(np.float64).eps};"
    + "double ___temperatureSafeDivide = (___temperatureTmp - 1.0 < 0) ? "
    + "___temperatureTmp - 1.0 - ___epsilon : "
    + "___temperatureTmp - 1.0 + ___epsilon;"
    + "___heat_transfer_coefficient_static_temperature = "
    + "abs(___temperatureTmp - 1.0) > ___epsilon ? "
    + "- heatFlux / ___temperatureSafeDivide :  1.0 / ___epsilon;",
    "solution.heat_transfer_coefficient_total_temperature": "double ___heat_transfer_coefficient_total_temperature;"
    + "double ___temperatureTmp = "
    + "primitiveVars[4] / (primitiveVars[0] * 1.0 / 1.4);"
    + "double ___temperatureTotal = 1.0 + (1.4 - 1.0) / 2.0 * MachRef * MachRef;"
    + f"double ___epsilon = {np.finfo(np.float64).eps};"
    + "double ___temperatureSafeDivide = (___temperatureTmp - ___temperatureTotal < 0) ? "
    + "___temperatureTmp - ___temperatureTotal - ___epsilon : "
    + "___temperatureTmp - ___temperatureTotal + ___epsilon;"
    + "___heat_transfer_coefficient_total_temperature = "
    + "abs(___temperatureTmp - ___temperatureTotal) > ___epsilon ? "
    + "___temperatureTotal = - heatFlux / ___temperatureSafeDivide :  1.0 / ___epsilon;",
    "solution.wall_shear_stress_magnitude": "double ___wall_shear_stress_magnitude;"
    + "___wall_shear_stress_magnitude = magnitude(wallShearStress);",
}
