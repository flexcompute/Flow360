from flow360.component.simulation.user_code import SolverVariable

mut = SolverVariable(name="fl.mut", value=float("NaN"))  # Turbulent viscosity
mu = SolverVariable(name="fl.mu", value=float("NaN"))  # Laminar viscosity
solutionNavierStokes = SolverVariable(
    name="fl.solutionNavierStokes", value=float("NaN")
)  # Solution for N-S equation in conservative form
residualNavierStokes = SolverVariable(
    name="fl.residualNavierStokes", value=float("NaN")
)  # Residual for N-S equation in conservative form
solutionTurbulence = SolverVariable(
    name="fl.solutionTurbulence", value=float("NaN")
)  # Solution for turbulence model
residualTurbulence = SolverVariable(
    name="fl.residualTurbulence", value=float("NaN")
)  # Residual for turbulence model
kOmega = SolverVariable(
    name="fl.kOmega", value=float("NaN")
)  # Effectively solutionTurbulence when using SST model
nuHat = SolverVariable(
    name="fl.nuHat", value=float("NaN")
)  # Effectively solutionTurbulence when using SA model
solutionTransition = SolverVariable(
    name="fl.solutionTransition", value=float("NaN")
)  # Solution for transition model
residualTransition = SolverVariable(
    name="fl.residualTransition", value=float("NaN")
)  # Residual for transition model
solutionHeatSolver = SolverVariable(
    name="fl.solutionHeatSolver", value=float("NaN")
)  # Solution for heat equation
residualHeatSolver = SolverVariable(
    name="fl.residualHeatSolver", value=float("NaN")
)  # Residual for heat equation
coordinate = SolverVariable(name="fl.coordinate", value=float("NaN"))  # Grid coordinates

physicalStep = SolverVariable(
    name="fl.physicalStep", value=float("NaN")
)  # Physical time step, starting from 0
pseudoStep = SolverVariable(
    name="fl.pseudoStep", value=float("NaN")
)  # Pseudo time step within physical time step
timeStepSize = SolverVariable(name="fl.timeStepSize", value=float("NaN"))  # Physical time step size
alphaAngle = SolverVariable(
    name="fl.alphaAngle", value=float("NaN")
)  # Alpha angle specified in freestream
betaAngle = SolverVariable(
    name="fl.betaAngle", value=float("NaN")
)  # Beta angle specified in freestream
pressureFreestream = SolverVariable(
    name="fl.pressureFreestream", value=float("NaN")
)  # Freestream reference pressure (1.0/1.4)
momentLengthX = SolverVariable(
    name="fl.momentLengthX", value=float("NaN")
)  # X component of momentLength
momentLengthY = SolverVariable(
    name="fl.momentLengthY", value=float("NaN")
)  # Y component of momentLength
momentLengthZ = SolverVariable(
    name="fl.momentLengthZ", value=float("NaN")
)  # Z component of momentLength
momentCenterX = SolverVariable(
    name="fl.momentCenterX", value=float("NaN")
)  # X component of momentCenter
momentCenterY = SolverVariable(
    name="fl.momentCenterY", value=float("NaN")
)  # Y component of momentCenter
momentCenterZ = SolverVariable(
    name="fl.momentCenterZ", value=float("NaN")
)  # Z component of momentCenter

bet_thrust = SolverVariable(name="fl.bet_thrust", value=float("NaN"))  # Thrust force for BET disk
bet_torque = SolverVariable(name="fl.bet_torque", value=float("NaN"))  # Torque for BET disk
bet_omega = SolverVariable(name="fl.bet_omega", value=float("NaN"))  # Rotation speed for BET disk
CD = SolverVariable(name="fl.CD", value=float("NaN"))  # Drag coefficient on patch
CL = SolverVariable(name="fl.CL", value=float("NaN"))  # Lift coefficient on patch
forceX = SolverVariable(name="fl.forceX", value=float("NaN"))  # Total force in X direction
forceY = SolverVariable(name="fl.forceY", value=float("NaN"))  # Total force in Y direction
forceZ = SolverVariable(name="fl.forceZ", value=float("NaN"))  # Total force in Z direction
momentX = SolverVariable(name="fl.momentX", value=float("NaN"))  # Total moment in X direction
momentY = SolverVariable(name="fl.momentY", value=float("NaN"))  # Total moment in Y direction
momentZ = SolverVariable(name="fl.momentZ", value=float("NaN"))  # Total moment in Z direction
nodeNormals = SolverVariable(name="fl.nodeNormals", value=float("NaN"))  # Normal vector of patch
theta = SolverVariable(name="fl.theta", value=float("NaN"))  # Rotation angle of volume zone
omega = SolverVariable(name="fl.omega", value=float("NaN"))  # Rotation speed of volume zone
omegaDot = SolverVariable(
    name="fl.omegaDot", value=float("NaN")
)  # Rotation acceleration of volume zone
wallFunctionMetric = SolverVariable(
    name="fl.wallFunctionMetric", value=float("NaN")
)  # Wall model quality indicator
wallShearStress = SolverVariable(
    name="fl.wallShearStress", value=float("NaN")
)  # Wall viscous shear stress
yPlus = SolverVariable(name="fl.yPlus", value=float("NaN"))  # Non-dimensional wall distance
