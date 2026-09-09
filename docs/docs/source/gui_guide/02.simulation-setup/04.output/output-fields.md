# Available Output Fields

*This section provides a comprehensive overview of all output fields (flow variables) available in Flow360 simulations, organized by their availability across different output types.*

> **Note:** All output fields are non-dimensional by default unless otherwise specified. See [Scaling Values and Nondimensionalization](./scaling-values.md) for information on converting to dimensional values.

---

## Universal Fields

*These fields are available for all output types.*

| *Field Name* | *Description* | *Units* |
|-------------|---------------|---------|
| `Cp` | Coefficient of pressure | Non-dimensional |
| `Cpt` | Coefficient of total pressure | Non-dimensional |
| `gradW` | Gradient of primitive solution | Non-dimensional |
| `kOmega` | k and omega (turbulence variables) | Non-dimensional |
| `Mach` | Mach number | Non-dimensional |
| `mut` | Turbulent viscosity | Non-dimensional |
| `mutRatio` | Turbulent viscosity and freestream dynamic viscosity ratio | Non-dimensional |
| `nuHat` | Spalart-Almaras variable | Non-dimensional |
| `primitiveVars` | Density, velocities (u,v,w), and pressure | Non-dimensional |
| `qcriterion` | Q criterion for vortex identification | Non-dimensional |
| `residualNavierStokes` | Navier-Stokes residual | Non-dimensional |
| `residualTransition` | Transition residual | Non-dimensional |
| `residualTurbulence` | Turbulence residual | Non-dimensional |
| `s` | Entropy | Non-dimensional |
| `solutionNavierStokes` | Navier-Stokes solution | Non-dimensional |
| `solutionTransition` | Transition solution | Non-dimensional |
| `solutionTurbulence` | Turbulence solution | Non-dimensional |
| `T` | Temperature | Non-dimensional |
| `velocity` | Velocity vector | Non-dimensional |
| `velocity_magnitude` | Magnitude of velocity vector | Non-dimensional |
| `pressure` | Pressure | Non-dimensional |
| `vorticity` | Vorticity | Non-dimensional |
| `vorticityMagnitude` | Vorticity magnitude | Non-dimensional |
| `wallDistance` | Wall distance | Grid unit length |
| `numericalDissipationFactor` | Numerical dissipation factor sensor | Non-dimensional |
| `residualHeatSolver` | Heat equation residual | Non-dimensional |
| `VelocityRelative` | Velocity with respect to non-inertial frame | Non-dimensional |
| `lowMachPreconditionerSensor` | Low-Mach preconditioner factor | Non-dimensional |

### VelocityRelative

This is the relative velocity with respect to the volume zone reference frame. In a rotational domain, the absolute velocity, $\overrightarrow{\boldsymbol{U}}_\text{absolute}$, of each fluid element could be treated as the summation of a relative velocity, $\overrightarrow{\boldsymbol{U}}_\text{relative}$, and a velocity due to the rotating frame, $\overrightarrow{\Omega}\times \overrightarrow{r}$. The "VelocityRelative" means the $\overrightarrow{\boldsymbol{U}}_\text{relative}$:

$$\overrightarrow{\boldsymbol{U}}_\text{absolute} = \overrightarrow{\boldsymbol{U}}_\text{relative} + \overrightarrow{\Omega}\times \overrightarrow{r}$$

It should be noted that the relative velocity is zero on no-slip walls that are part of the rotating frame (i.e., rotating walls) within rotational blocks. When a wall function is used, this velocity is near zero.

---

## Volume and Slice Specific Fields

*These fields are available only for Volume Output and Slice Output types.*

| *Field Name* | *Description* | *Units* |
|-------------|---------------|---------|
| `betMetrics` | BET Metrics | Non-dimensional |
| `betMetricsPerDisk` | BET Metrics per Disk | Non-dimensional |
| `linearResidualNavierStokes` | Linear residual of Navier-Stokes solver | Non-dimensional |
| `linearResidualTurbulence` | Linear residual of turbulence solver | Non-dimensional |
| `linearResidualTransition` | Linear residual of transition solver | Non-dimensional |
| `SpalartAllmaras_hybridModel` | Hybrid RANS-LES output for Spalart-Allmaras solver (supports both DDES and ZDES) | Non-dimensional |
| `kOmegaSST_hybridModel` | Hybrid RANS-LES output for kOmegaSST solver (supports both DDES and ZDES) | Non-dimensional |
| `localCFL` | Local CFL number | Non-dimensional |
| `pressureTimeDerivative` | Time derivative of static pressure. Unsteady simulations only. Also available for Volume Probe Output. In the WebUI it appears as `pressureTimeDerivative(dp/dt)` on the time-averaging volume, slice, surface and probe panels; the instantaneous outputs accept it through the Python API only | Non-dimensional |

### BET Metrics Output Variables

The `betMetrics` and `betMetricsPerDisk` output fields provide Blade Element Theory (BET) metrics for analyzing rotor and propeller performance. These fields are available when using BET models in volume zones. The `betMetrics` field includes data from all BET disks with possible overlapping, while `betMetricsPerDisk` provides separate outputs for each disk to avoid overlap.

The following variables are included in the betMetrics output:

1. **`VelocityRelative`** – Relative velocity with respect to the rotating reference frame (non-dimensional).

2. **`AlphaRadians`** – Local angle of attack in radians.

3. **`CfAxial`** – Axial aerodynamic force coefficient.

4. **`CfCircumferential`** – Circumferential aerodynamic force coefficient.

5. **`TipLossFactor`** – Factor to model the effect of blade tip.

6. **`LocalSolidityIntegralWeight`** – Local solidity multiplied by the integral weight.

### Hybrid RANS-LES Output Variables

The `SpalartAllmaras_hybridModel` and `kOmegaSST_hybridModel` output fields provide diagnostic variables for hybrid RANS-LES simulations. The specific variables included depend on whether you're using **DDES** (Delayed Detached Eddy Simulation) or **ZDES** (Zonal Detached Eddy Simulation) as the shielding function.

```{important}
Each field requires the matching turbulence model with hybrid RANS-LES enabled: `SpalartAllmaras_hybridModel` needs the Spalart-Allmaras model, and `kOmegaSST_hybridModel` needs the kOmegaSST model. Requesting one without the matching model and an active hybrid setting is rejected during validation, and the message names the output section at fault.
```

#### DDES Variables (when `shielding_function="DDES"`)

When using DDES, the hybrid model output includes five key variables:

1. **`f_d`** – The shielding function that delineates the RANS and LES regions. When `f_d` = 0, the RANS model is fully applied; when `f_d` = 1, the LES model is used. Intermediate values represent a smooth transition between the two regimes.

2. **`r_d`** – A modified ratio of the modeled length scale to the wall distance, from which `f_d` is derived.

3. **`DDES_lengthRANS`** – The wall distance from the computational cell to the nearest solid boundary.

4. **`DDES_lengthScale`** – The characteristic DES length scale: $\tilde{d} \equiv d - f_d \max(0, d - C_{DES}*\Delta)$

5. **`DDES_lengthLES`** – The characteristic LES length scale: $C_{DES}*\Delta$

Among these variables, `f_d` is the most significant, as it enables users to identify and visualize the regions dominated by RANS and DES behavior within the computational domain.

#### ZDES Variables (when `shielding_function="ZDES"`)

When using ZDES, the hybrid model output includes four key variables:

1. **`ZDES_fp`** – The enhanced shielding function that determines whether RANS or LES is used. When `ZDES_fp` = 0, RANS is active; when `ZDES_fp` = 1, LES is active. This function is computed from `ZDES_fd`, `ZDES_fR`, and `ZDES_fp2`.

2. **`ZDES_fd`** – Original DDES shielding function used in computing `ZDES_fp`.

3. **`ZDES_fR`** – Component used in computing `ZDES_fp`. This is included to disable or inhibit the second shielding function in regions where vorticity magnitude is increasing away from walls - this is designed to disable the secondary shielding function where a shear layer is detected above a wall.

4. **`ZDES_fp2`** – Causes the model to revert to RANS mode in the outer portion of boundary layers, used in computing `ZDES_fp`.

---

## Surface Specific Fields

*These fields are available only for Surface Output and Surface Probe Output types.*

| *Field Name* | *Description* | *Units* |
|-------------|---------------|---------|
| `CfVec` | Skin friction coefficient vector | Non-dimensional |
| `Cf` | Magnitude of skin friction coefficient | Non-dimensional |
| `heatFlux` | Non-dimensional heat flux | Non-dimensional |
| `nodeNormals` | Wall normals | Non-dimensional |
| `nodeForcesPerUnitArea` | Forces per unit area | Non-dimensional |
| `yPlus` | Non-dimensional wall distance | Non-dimensional |
| `wallFunctionMetric` | Wall function metrics | Non-dimensional |
| `heatTransferCoefficientStaticTemperature` | Surface heat transfer coefficient (static temperature as reference) | Non-dimensional |
| `heatTransferCoefficientTotalTemperature` | Surface heat transfer coefficient (total temperature as reference) | Non-dimensional |
| `wall_shear_stress_magnitude` | Wall shear stress magnitude | Non-dimensional |
| `wall_shear_stress_magnitude_pa` | Wall shear stress magnitude | **Pascals (Pa)** - Available since version 25.2 |
| `pressureTimeDerivative` | Time derivative of static pressure. Unsteady simulations only. In the WebUI it appears as `pressureTimeDerivative(dp/dt)` on the time-averaging surface panel; the instantaneous surface output accepts it through the Python API only | Non-dimensional |

---

## Isosurface Specific Fields

*These fields are available only for Isosurface Output types.*

## Visualization Tips
Isosurface outputs support all universal fields listed above. The most commonly used fields for isosurface visualization are:

- `qcriterion` - For vortex identification
- `Mach` - For shock wave visualization
- `pressure` - For pressure-based isosurfaces
- `Cpt` - For total pressure loss visualization

---

## Custom Variables

*User-defined expressions with dimensions. These can be created using the Variable Settings tool or Python API.*

By default, the following expressions are available:

| *Variable Name* | *Expression* | *Description* |
|----------------|--------------|---------------|
| `velocity_with_units` | `solution.velocity` | Velocity in physical units |
| `velocity_magnitude_with_units` | `math.magnitude(solution.velocity)` | Velocity magnitude in physical units |
| `pressure_with_units` | `solution.pressure` | Pressure in physical units |
| `wall_shear_stress_magnitude_with_units` | `solution.wall_shear_stress_magnitude` | Wall shear stress magnitude in physical units |

> **Note:** You can create additional custom variables using the [Variable Settings](../../05.tools/02.variable-settings.md) tool or through the Python API. Custom variables can access multiple solver variables and undergo mathematical operations.

---

## Related Documentation

- [Scaling Values and Nondimensionalization](./scaling-values.md) - Learn how to convert non-dimensional values to physical units
- [Variable Settings](../../05.tools/02.variable-settings.md) - Create custom output variables
- [Reference Dimensions](./reference-dimensions.md) - Configure reference dimensions for coefficient calculations
