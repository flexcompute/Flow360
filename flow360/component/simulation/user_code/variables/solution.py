"""Solution variables of Flow360"""

import unyt as u

from flow360.component.simulation.user_code.core.types import SolverVariable

# pylint:disable = no-member
mut = SolverVariable(
    name="solution.mut", value=float("NaN") * u.kg / u.m / u.s, solver_name="mut"
)  # Turbulent viscosity
mu = SolverVariable(name="solution.mu", value=float("NaN") * u.kg / u.m / u.s)  # Laminar viscosity

solutionNavierStokes = SolverVariable(
    name="solution.solutionNavierStokes", value=float("NaN")
)  # Solution for N-S equation in conservative form
residualNavierStokes = SolverVariable(
    name="solution.residualNavierStokes", value=float("NaN")
)  # Residual for N-S equation in conservative form
solutionTurbulence = SolverVariable(
    name="solution.solutionTurbulence", value=float("NaN")
)  # Solution for turbulence model
residualTurbulence = SolverVariable(
    name="solution.residualTurbulence", value=float("NaN")
)  # Residual for turbulence model
kOmega = SolverVariable(
    name="solution.kOmega", value=float("NaN")
)  # Effectively solutionTurbulence when using SST model
nuHat = SolverVariable(
    name="solution.nuHat", value=float("NaN")
)  # Effectively solutionTurbulence when using SA model
solutionTransition = SolverVariable(
    name="solution.solutionTransition", value=float("NaN")
)  # Solution for transition model
residualTransition = SolverVariable(
    name="solution.residualTransition", value=float("NaN")
)  # Residual for transition model
solutionHeatSolver = SolverVariable(
    name="solution.solutionHeatSolver", value=float("NaN")
)  # Solution for heat equation
residualHeatSolver = SolverVariable(
    name="solution.residualHeatSolver", value=float("NaN")
)  # Residual for heat equation

coordinate = SolverVariable(
    name="solution.coordinate",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m,
)  # Grid coordinates

velocity = SolverVariable(
    name="solution.velocity",
    value=[float("NaN"), float("NaN"), float("NaN")] * u.m / u.s,
)

bet_thrust = SolverVariable(
    name="solution.bet_thrust", value=float("NaN")
)  # Thrust force for BET disk
bet_torque = SolverVariable(name="solution.bet_torque", value=float("NaN"))  # Torque for BET disk
bet_omega = SolverVariable(
    name="solution.bet_omega", value=float("NaN")
)  # Rotation speed for BET disk
CD = SolverVariable(name="solution.CD", value=float("NaN"))  # Drag coefficient on patch
CL = SolverVariable(name="solution.CL", value=float("NaN"))  # Lift coefficient on patch
forceX = SolverVariable(name="solution.forceX", value=float("NaN"))  # Total force in X direction
forceY = SolverVariable(name="solution.forceY", value=float("NaN"))  # Total force in Y direction
forceZ = SolverVariable(name="solution.forceZ", value=float("NaN"))  # Total force in Z direction
momentX = SolverVariable(name="solution.momentX", value=float("NaN"))  # Total moment in X direction
momentY = SolverVariable(name="solution.momentY", value=float("NaN"))  # Total moment in Y direction
momentZ = SolverVariable(name="solution.momentZ", value=float("NaN"))  # Total moment in Z direction
nodeNormals = SolverVariable(
    name="solution.nodeNormals", value=float("NaN")
)  # Normal vector of patch
wallFunctionMetric = SolverVariable(
    name="solution.wallFunctionMetric", value=float("NaN")
)  # Wall model quality indicator
wallShearStress = SolverVariable(
    name="solution.wallShearStress", value=float("NaN")
)  # Wall viscous shear stress
yPlus = SolverVariable(name="solution.yPlus", value=float("NaN"))  # Non-dimensional wall distance
