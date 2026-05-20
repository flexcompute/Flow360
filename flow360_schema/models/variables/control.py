"""Control variables of Flow360"""

import unyt as u

from flow360_schema.framework.expression.variable import SolverVariable

MachRef = SolverVariable(  # type: ignore[call-arg]
    name="control.MachRef",
    value=float("NaN"),
    solver_name="machRef",
    variable_type="Scalar",
)  # Reference mach specified by the user
Tref = SolverVariable(  # type: ignore[call-arg]
    name="control.Tref", value=float("NaN") * u.K, variable_type="Scalar"
)  # Temperature specified by the user
t = SolverVariable(  # type: ignore[call-arg]
    name="control.t", value=float("NaN") * u.s, variable_type="Scalar"
)  # Physical time
physicalStep = SolverVariable(  # type: ignore[call-arg]
    name="control.physicalStep", value=float("NaN"), variable_type="Scalar"
)  # Physical time step, starting from 0
pseudoStep = SolverVariable(  # type: ignore[call-arg]
    name="control.pseudoStep", value=float("NaN"), variable_type="Scalar"
)  # Pseudo time step within physical time step
timeStepSize = SolverVariable(  # type: ignore[call-arg]
    name="control.timeStepSize", value=float("NaN") * u.s, variable_type="Scalar"
)  # Physical time step size
alphaAngle = SolverVariable(  # type: ignore[call-arg]
    name="control.alphaAngle", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Alpha angle specified in freestream
betaAngle = SolverVariable(  # type: ignore[call-arg]
    name="control.betaAngle", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Beta angle specified in freestream
pressureFreestream = SolverVariable(  # type: ignore[call-arg]
    name="control.pressureFreestream", value=float("NaN") * u.Pa, variable_type="Scalar"
)  # Freestream reference pressure (1.0/1.4)
momentLengthX = SolverVariable(  # type: ignore[call-arg]
    name="control.momentLengthX", value=float("NaN") * u.m, variable_type="Scalar"
)  # X component of momentLength
momentLengthY = SolverVariable(  # type: ignore[call-arg]
    name="control.momentLengthY", value=float("NaN") * u.m, variable_type="Scalar"
)  # Y component of momentLength
momentLengthZ = SolverVariable(  # type: ignore[call-arg]
    name="control.momentLengthZ", value=float("NaN") * u.m, variable_type="Scalar"
)  # Z component of momentLength
momentCenterX = SolverVariable(  # type: ignore[call-arg]
    name="control.momentCenterX", value=float("NaN") * u.m, variable_type="Scalar"
)  # X component of momentCenter
momentCenterY = SolverVariable(  # type: ignore[call-arg]
    name="control.momentCenterY", value=float("NaN") * u.m, variable_type="Scalar"
)  # Y component of momentCenter
momentCenterZ = SolverVariable(  # type: ignore[call-arg]
    name="control.momentCenterZ", value=float("NaN") * u.m, variable_type="Scalar"
)  # Z component of momentCenter
theta = SolverVariable(  # type: ignore[call-arg]
    name="control.theta", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Rotation angle of volume zone
omega = SolverVariable(  # type: ignore[call-arg]
    name="control.omega", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Rotation speed of volume zone
omegaDot = SolverVariable(  # type: ignore[call-arg]
    name="control.omegaDot", value=float("NaN") * u.rad / u.s, variable_type="Scalar"
)  # Rotation acceleration of volume zone
