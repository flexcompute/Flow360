"""Control variables of Flow360"""

from flow360.component.simulation import units as u
from flow360.component.simulation.user_code.core.types import SolverVariable

# pylint:disable=no-member
MachRef = SolverVariable(
    name="control.MachRef",
    value=float("NaN"),
    solver_name="machRef",
    variable_type="Scalar",
)  # Reference mach specified by the user
Tref = SolverVariable(
    name="control.Tref", value=float("NaN") * u.K, variable_type="Scalar"
)  # Temperature specified by the user
t = SolverVariable(
    name="control.t", value=float("NaN") * u.s, variable_type="Scalar"
)  # Physical time
physicalStep = SolverVariable(
    name="control.physicalStep", value=float("NaN"), variable_type="Scalar"
)  # Physical time step, starting from 0
pseudoStep = SolverVariable(
    name="control.pseudoStep", value=float("NaN"), variable_type="Scalar"
)  # Pseudo time step within physical time step
timeStepSize = SolverVariable(
    name="control.timeStepSize", value=float("NaN") * u.s, variable_type="Scalar"
)  # Physical time step size
alphaAngle = SolverVariable(
    name="control.alphaAngle", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Alpha angle specified in freestream
betaAngle = SolverVariable(
    name="control.betaAngle", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Beta angle specified in freestream
pressureFreestream = SolverVariable(
    name="control.pressureFreestream", value=float("NaN") * u.Pa, variable_type="Scalar"
)  # Freestream reference pressure (1.0/1.4)
momentLengthX = SolverVariable(
    name="control.momentLengthX", value=float("NaN") * u.m, variable_type="Scalar"
)  # X component of momentLength
momentLengthY = SolverVariable(
    name="control.momentLengthY", value=float("NaN") * u.m, variable_type="Scalar"
)  # Y component of momentLength
momentLengthZ = SolverVariable(
    name="control.momentLengthZ", value=float("NaN") * u.m, variable_type="Scalar"
)  # Z component of momentLength
momentCenterX = SolverVariable(
    name="control.momentCenterX", value=float("NaN") * u.m, variable_type="Scalar"
)  # X component of momentCenter
momentCenterY = SolverVariable(
    name="control.momentCenterY", value=float("NaN") * u.m, variable_type="Scalar"
)  # Y component of momentCenter
momentCenterZ = SolverVariable(
    name="control.momentCenterZ", value=float("NaN") * u.m, variable_type="Scalar"
)  # Z component of momentCenter
theta = SolverVariable(
    name="control.theta", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Rotation angle of volume zone
omega = SolverVariable(
    name="control.omega", value=float("NaN") * u.rad, variable_type="Scalar"
)  # Rotation speed of volume zone
omegaDot = SolverVariable(
    name="control.omegaDot", value=float("NaN") * u.rad / u.s, variable_type="Scalar"
)  # Rotation acceleration of volume zone
