"""Utilities for user expression translation."""

import numpy as np

udf_prepending_code = {
    "solution.Cp": {
        "declaration": "double Cp;",
        "computation": "Cp = (primitiveVars[4] - pressureFreestream) / (0.5 * MachRef * MachRef);",
    },
    "solution.Cpt": {
        "declaration": "double Cpt;",
        "computation": "double MachTmp = sqrt(primitiveVars[1] * primitiveVars[1] + "
        + "primitiveVars[2] * primitiveVars[2] + primitiveVars[3] * primitiveVars[3]) / "
        + "sqrt(1.4 * primitiveVars[4] / primitiveVars[0]);"
        + "Cpt = (1.4 * primitiveVars[4] * pow(1.0 + (1.4 - 1.0) / 2. * MachTmp * MachTmp,"
        + "1.4 / (1.4 - 1.0)) - pow(1.0 + (1.4 - 1.0) / 2. * MachRef * MachRef,"
        + "1.4 / (1.4 - 1.0))) / (0.5 * 1.4 * MachRef * MachRef);",
    },
    "solution.grad_density": {
        "declaration": "double gradDensity[3];",
        "computation": " gradDensity[0] = gradPrimitive[0][0];"
        + "gradDensity[1] = gradPrimitive[0][1];"
        + "gradDensity[2] = gradPrimitive[0][2];",
    },
    "solution.grad_u": {
        "declaration": "double gradVelocityX[3];",
        "computation": "gradVelocityX[0] = gradPrimitive[1][0] * velocityScale;"
        + "gradVelocityX[1] = gradPrimitive[1][1] * velocityScale;"
        + "gradVelocityX[2] = gradPrimitive[1][2] * velocityScale;",
    },
    "solution.grad_v": {
        "declaration": "double gradVelocityY[3];",
        "computation": "gradVelocityY[0] = gradPrimitive[2][0] * velocityScale;"
        + "gradVelocityY[1] = gradPrimitive[2][1] * velocityScale;"
        + "gradVelocityY[2] = gradPrimitive[2][2] * velocityScale;",
    },
    "solution.grad_w": {
        "declaration": "double gradVelocityZ[3];",
        "computation": "gradVelocityZ[0] = gradPrimitive[3][0] * velocityScale;"
        + "gradVelocityZ[1] = gradPrimitive[3][1] * velocityScale;"
        + "gradVelocityZ[2] = gradPrimitive[3][2] * velocityScale;",
    },
    "solution.grad_pressure": {
        "declaration": "double gradPressure[3];",
        "computation": "gradPressure[0] = gradPrimitive[4][0]; "
        + "gradPressure[1] = gradPrimitive[4][1]; "
        + "gradPressure[2] = gradPrimitive[4][2];",
    },
    "solution.Mach": {
        "declaration": "double Mach;",
        "computation": "Mach = usingLiquidAsMaterial ? 0 : "
        + "sqrt(primitiveVars[1] * primitiveVars[1] + "
        + "primitiveVars[2] * primitiveVars[2] + "
        + "primitiveVars[3] * primitiveVars[3]) / "
        + "sqrt(1.4 * primitiveVars[4] / primitiveVars[0]);",
    },
    "solution.mut_ratio": {
        "declaration": "double mutRatio;",
        "computation": "mutRatio = mut / mu;",
    },
    "solution.nu_hat": {
        "declaration": "double nuHat;",
        "computation": "nuHat = SpalartAllmaras_solution * velocityScale;",
    },
    "solution.turbulence_kinetic_energy": {
        "declaration": "double turbulenceKineticEnergy;",
        "computation": "turbulenceKineticEnergy = kOmegaSST_solution[0] * pow(velocityScale, 2);",
    },
    "solution.specific_rate_of_dissipation": {
        "declaration": "double specificRateOfDissipation;",
        "computation": "specificRateOfDissipation = kOmegaSST_solution[1] * velocityScale;",
    },
    "solution.velocity": {
        "declaration": "double velocity[3];",
        "computation": "velocity[0] = primitiveVars[1] * velocityScale;"
        + "velocity[1] = primitiveVars[2] * velocityScale;"
        + "velocity[2] = primitiveVars[3] * velocityScale;",
    },
    "solution.velocity_magnitude": {
        "declaration": "double velocityMagnitude;",
        "computation": "double velocityTmp[3];velocityTmp[0] = primitiveVars[1] * velocityScale;"
        + "velocityTmp[1] = primitiveVars[2] * velocityScale;"
        + "velocityTmp[2] = primitiveVars[3] * velocityScale;"
        + "velocityMagnitude = magnitude(velocityTmp);",
    },
    "solution.qcriterion": {
        "declaration": "double qcriterion;",
        "computation": "double ux = gradPrimitive[1][0];"
        + "double uy = gradPrimitive[1][1];"
        + "double uz = gradPrimitive[1][2];"
        + "double vx = gradPrimitive[2][0];"
        + "double vy = gradPrimitive[2][1];"
        + "double vz = gradPrimitive[2][2];"
        + "double wx = gradPrimitive[3][0];"
        + "double wy = gradPrimitive[3][1];"
        + "double wz = gradPrimitive[3][2];"
        + "double str11 = ux;"
        + "double str22 = vy;"
        + "double str33 = wz;"
        + "double str12 = 0.5 * (uy + vx);"
        + "double str13 = 0.5 * (uz + wx);"
        + "double str23 = 0.5 * (vz + wy);"
        + "double str_norm = str11 * str11 + str22 * str22 + str33 * str33 + "
        + "2 * (str12 * str12) + 2 * (str13 * str13) + 2 * (str23 * str23);"
        + "double omg12 = 0.5 * (uy - vx);"
        + "double omg13 = 0.5 * (uz - wx);"
        + "double omg23 = 0.5 * (vz - wy);"
        + "double omg_norm = 2 * (omg12 * omg12) + 2 * (omg13 * omg13) + 2 * (omg23 * omg23);"
        + "qcriterion = 0.5 * (omg_norm - str_norm) * (velocityScale * velocityScale);",
    },
    "solution.entropy": {
        "declaration": "double entropy;",
        "computation": "entropy = log(primitiveVars[4] / (1.0 / 1.4) / pow(primitiveVars[0], 1.4));",
    },
    "solution.temperature": {
        "declaration": "double temperature;",
        "computation": f"double epsilon = {np.finfo(np.float64).eps};"
        "temperature = (primitiveVars[0] < epsilon && HeatEquation_solution != nullptr) ? "
        "HeatEquation_solution[0] : primitiveVars[4] / (primitiveVars[0] * (1.0 / 1.4));",
    },
    "solution.vorticity": {
        "declaration": "double vorticity[3];",
        "computation": "vorticity[0] = (gradPrimitive[3][1] - gradPrimitive[2][2]) * velocityScale;"
        + "vorticity[1] = (gradPrimitive[1][2] - gradPrimitive[3][0]) * velocityScale;"
        + "vorticity[2] = (gradPrimitive[2][0] - gradPrimitive[1][1]) * velocityScale;",
    },
    "solution.vorticity_magnitude": {
        "declaration": "double vorticityMagnitude;",
        "computation": "double vorticityTmp[3];"
        + "vorticityTmp[0] = (gradPrimitive[3][1] - gradPrimitive[2][2]) * velocityScale;"
        + "vorticityTmp[1] = (gradPrimitive[1][2] - gradPrimitive[3][0]) * velocityScale;"
        + "vorticityTmp[2] = (gradPrimitive[2][0] - gradPrimitive[1][1]) * velocityScale;"
        + "vorticityMagnitude = magnitude(vorticityTmp);",
    },
    "solution.CfVec": {
        "declaration": "double CfVec[3];",
        "computation": "for (int i = 0; i < 3; i++)"
        + "{CfVec[i] = wallShearStress[i] / (0.5 * MachRef * MachRef);}",
    },
    "solution.Cf": {
        "declaration": "double Cf;",
        "computation": "Cf = magnitude(wallShearStress) / (0.5 * MachRef * MachRef);",
    },
    "solution.node_forces_per_unit_area": {
        "declaration": "double nodeForcesPerUnitArea[3];",
        "computation": "double normalMag = magnitude(nodeNormals);"
        + "for (int i = 0; i < 3; i++){nodeForcesPerUnitArea[i] = "
        + "((primitiveVars[4] - pressureFreestream) * nodeNormals[i] / normalMag + wallViscousStress[i])"
        + " * (velocityScale * velocityScale);}",
    },
    "solution.heat_transfer_coefficient_static_temperature": {
        "declaration": "double heatTransferCoefficientStaticTemperature;",
        "computation": "double temperatureTmp = "
        + "primitiveVars[4] / (primitiveVars[0] * 1.0 / 1.4);"
        + f"double epsilon = {np.finfo(np.float64).eps};"
        + "double temperatureSafeDivide = (temperatureTmp - 1.0 < 0) ? "
        + "temperatureTmp - 1.0 - epsilon : "
        + "temperatureTmp - 1.0 + epsilon;"
        + "heatTransferCoefficientStaticTemperature = "
        + "abs(temperatureTmp - 1.0) > epsilon ? "
        + "- wallHeatFlux / temperatureSafeDivide :  1.0 / epsilon;",
    },
    "solution.heat_transfer_coefficient_total_temperature": {
        "declaration": "double heatTransferCoefficientTotalTemperature;",
        "computation": "double temperatureTmp = "
        + "primitiveVars[4] / (primitiveVars[0] * 1.0 / 1.4);"
        + "double temperatureTotal = 1.0 + (1.4 - 1.0) / 2.0 * MachRef * MachRef;"
        + f"double epsilon = {np.finfo(np.float64).eps};"
        + "double temperatureSafeDivide = (temperatureTmp - temperatureTotal < 0) ? "
        + "temperatureTmp - temperatureTotal - epsilon : "
        + "temperatureTmp - temperatureTotal + epsilon;"
        + "double heatTransferCoefficientTotalTemperature = "
        + "abs(temperatureTmp - temperatureTotal) > epsilon ? "
        + "temperatureTotal = - wallHeatFlux / temperatureSafeDivide :  1.0 / epsilon;",
    },
    "solution.wall_shear_stress_magnitude": {
        "declaration": "double wallShearStressMagnitude;",
        "computation": "wallShearStressMagnitude = magnitude(wallShearStress);",
    },
}
