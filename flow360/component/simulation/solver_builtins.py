from flow360.component.simulation.user_code import Variable

mut = Variable(name="fl.mut", value=float("NaN")) # Turbulent viscosity
mu = Variable(name="fl.mu", value=float("NaN")) # Laminar viscosity
solutionNavierStokes = Variable(name="fl.solutionNavierStokes", value=float("NaN")) # Solution for N-S equation in conservative form
residualNavierStokes = Variable(name="fl.residualNavierStokes", value=float("NaN")) # Residual for N-S equation in conservative form
solutionTurbulence = Variable(name="fl.solutionTurbulence", value=float("NaN")) # Solution for turbulence model
residualTurbulence = Variable(name="fl.residualTurbulence", value=float("NaN")) # Residual for turbulence model
kOmega = Variable(name="fl.kOmega", value=float("NaN")) # Effectively solutionTurbulence when using SST model
nuHat = Variable(name="fl.nuHat", value=float("NaN")) # Effectively solutionTurbulence when using SA model
solutionTransition = Variable(name="fl.solutionTransition", value=float("NaN")) # Solution for transition model
residualTransition = Variable(name="fl.residualTransition", value=float("NaN")) # Residual for transition model
solutionHeatSolver = Variable(name="fl.solutionHeatSolver", value=float("NaN")) # Solution for heat equation
residualHeatSolver = Variable(name="fl.residualHeatSolver", value=float("NaN")) # Residual for heat equation
coordinate = Variable(name="fl.coordinate", value=float("NaN")) # Grid coordinates

physicalStep = Variable(name="fl.physicalStep", value=float("NaN")) # Physical time step, starting from 0
pseudoStep = Variable(name="fl.pseudoStep", value=float("NaN")) # Pseudo time step within physical time step
timeStepSize = Variable(name="fl.timeStepSize", value=float("NaN")) # Physical time step size
alphaAngle = Variable(name="fl.alphaAngle", value=float("NaN")) # Alpha angle specified in freestream
betaAngle = Variable(name="fl.betaAngle", value=float("NaN")) # Beta angle specified in freestream
pressureFreestream = Variable(name="fl.pressureFreestream", value=float("NaN"))  # Freestream reference pressure (1.0/1.4)
momentLengthX = Variable(name="fl.momentLengthX", value=float("NaN")) # X component of momentLength
momentLengthY = Variable(name="fl.momentLengthY", value=float("NaN")) # Y component of momentLength
momentLengthZ = Variable(name="fl.momentLengthZ", value=float("NaN")) # Z component of momentLength
momentCenterX = Variable(name="fl.momentCenterX", value=float("NaN")) # X component of momentCenter
momentCenterY = Variable(name="fl.momentCenterY", value=float("NaN")) # Y component of momentCenter
momentCenterZ = Variable(name="fl.momentCenterZ", value=float("NaN")) # Z component of momentCenter

bet_thrust = Variable(name="fl.bet_thrust", value=float("NaN")) # Thrust force for BET disk
bet_torque = Variable(name="fl.bet_torque", value=float("NaN")) # Torque for BET disk
bet_omega = Variable(name="fl.bet_omega", value=float("NaN")) # Rotation speed for BET disk
CD = Variable(name="fl.CD", value=float("NaN")) # Drag coefficient on patch
CL = Variable(name="fl.CL", value=float("NaN")) # Lift coefficient on patch
forceX = Variable(name="fl.forceX", value=float("NaN")) # Total force in X direction
forceY = Variable(name="fl.forceY", value=float("NaN")) # Total force in Y direction
forceZ = Variable(name="fl.forceZ", value=float("NaN")) # Total force in Z direction
momentX = Variable(name="fl.momentX", value=float("NaN")) # Total moment in X direction
momentY = Variable(name="fl.momentY", value=float("NaN")) # Total moment in Y direction
momentZ = Variable(name="fl.momentZ", value=float("NaN")) # Total moment in Z direction
nodeNormals = Variable(name="fl.nodeNormals", value=float("NaN")) # Normal vector of patch
theta = Variable(name="fl.theta", value=float("NaN")) # Rotation angle of volume zone
omega = Variable(name="fl.omega", value=float("NaN")) # Rotation speed of volume zone
omegaDot = Variable(name="fl.omegaDot", value=float("NaN")) # Rotation acceleration of volume zone
wallFunctionMetric = Variable(name="fl.wallFunctionMetric", value=float("NaN")) # Wall model quality indicator
wallShearStress = Variable(name="fl.wallShearStress", value=float("NaN")) # Wall viscous shear stress
yPlus = Variable(name="fl.yPlus", value=float("NaN")) # Non-dimensional wall distance

