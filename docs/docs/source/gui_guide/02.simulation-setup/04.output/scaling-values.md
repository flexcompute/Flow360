# Scaling Values and Nondimensionalization

*This section explains how Flow360 nondimensionalizes output values and the reference scaling values used for converting between dimensional and non-dimensional quantities.*

> **Note:** Flow360 uses nondimensional values internally for numerical stability and consistency. When interpreting output data, it's important to understand how these values are nondimensionalized.

## Reference Values for Nondimensionalization

| *Property*   | *Reference Value for Nondimensionalization*   | *Examples in Flow360*                        |
|-------------|---------------------------------------------|----------------------------------------------|
| Length      | Grid Unit Length                            | wallDistance in volume and surface outputs   |
| Density     | Freestream Density ($ρ_∞$)                     | primitiveVars in all outputs                |
| Velocity    | Reference velocity scaling ($U_{scale}$)            | primitiveVars in all outputs                |
| Pressure    | $ρ_∞ \cdot U_{scale}^2$                           | primitiveVars, nodeForcesPerUnitArea        |
| Temperature | Freestream Temperature ($T_∞$)                 | T in volume outputs                         |
| Heat Flux   | $ρ_∞ \cdot U_{scale}^3$                           | heatFlux in surface outputs                 |
| Force (BET/AD/Porous Media) | $ρ_∞ \cdot U_{scale}^2 \cdot (Grid Unit Length)^2$ | Forces in BET/AD/Porous Media outputs       |
| Moment (BET/AD/Porous Media) | $ρ_∞ \cdot U_{scale}^2 \cdot (Grid Unit Length)^3$ | Moments in BET/AD/Porous Media outputs      |

### Reference Velocity Scaling

The reference velocity scaling ($U_{scale}$) is the reference velocity used for nondimensionalizing velocity-related variables (velocity fields, heat flux, angular speeds, etc.):

- For `AerospaceCondition`: $U_{scale} = C_∞$ (speed of sound)
- For `LiquidOperatingCondition`: $U_{scale}$ is the reference velocity (if set) or velocity magnitude (if reference velocity is not set)

> **Note**: It is important to distinguish $U_{scale}$ from $U_{ref}$ (reference velocity used for force and moment coefficients). While both may have the same value in some cases, they serve different purposes:
> - $U_{scale}$ is used for nondimensionalizing velocity-related variables (velocity fields, heat flux, volumetric heat sources, angular speeds, etc.)
> - $U_{ref}$ is the user-specified reference velocity used specifically for force and moment coefficients (CL, CD, etc.), skin friction coefficient (Cf), pressure coefficient (Cp), and total pressure coefficient (Cpt)

## Important Coefficients and Their Scaling

### Skin Friction Coefficient (Cf, CfVec)

The skin friction coefficient represents the wall shear stress nondimensionalized by the dynamic pressure:

* `CfVec` is the skin friction coefficient vector, showing both magnitude and direction
* `Cf` is the magnitude of that vector

To calculate the dimensional viscous stress on the wall:

$\tau_{wall} [\frac{N}{m^2}] = C_f \cdot \frac{1}{2}ρ_∞ \cdot U_{ref}^2$

where $U_{ref}$ is the reference velocity used for force and moment coefficients (set via `Mach` or `MachRef` parameters in the operating condition).

**Recommended Method**: For convenience, Flow360 provides the wall shear stress directly in physical units through the `wall_shear_stress_magnitude_with_units` field using the specified units in webUI. This is the recommended method to access wall shear stress in dimensional form without manual conversion.

`CfVec` is particularly useful for identifying boundary layer separation:
* Fully attached flow follows the surface along the streamwise direction
* Separated flow induces local recirculation
* Negative values of the streamwise component (e.g., `CfVecX` for flow in the x-direction) indicate boundary layer separation

### Pressure Coefficient (Cp)

The pressure coefficient represents the difference between local and freestream static pressure, normalized by dynamic pressure:

To calculate the dimensional pressure:

$p [\frac{N}{m^2}] = C_p \cdot \frac{1}{2}ρ_∞ \cdot U_{ref}^2 + p_∞$

where $U_{ref}$ is the reference velocity used for force and moment coefficients (set via `Mach` or `MachRef` parameters in the operating condition), and $p_∞$ is the ambient pressure (pressure at the farfield).

**Recommended Method**: For convenience, Flow360 provides the pressure directly in physical units through the `pressure_with_units` field using the specified units in webUI. This is the recommended method to access pressure in dimensional form without manual conversion.

### Total Pressure Coefficient (Cpt)

The total pressure coefficient is useful for identifying losses in the flow field:

Total pressure (or stagnation pressure) is the sum of static pressure, dynamic pressure, and gravitational head (often negligible). At stagnation points where velocity is zero, dynamic pressure becomes zero and total pressure equals static pressure.

To calculate the dimensional total pressure:

$p_t [\frac{N}{m^2}] = {C_p}_t \cdot \frac{1}{2}ρ_∞ \cdot U_{ref}^2 + {p_\infty}_t$

Total pressure coefficient is excellent for visualizing:
* Boundary layer development
* Regions of separation in the flow volume
* Wakes behind objects

### Q-Criterion

Q-criterion is used to identify vortical structures in the flow field. It represents the balance between rotation rate and strain rate in the flow.

* Positive values indicate areas where rotation dominates over strain (vortex cores)
* Higher values indicate stronger vortices
* The default isosurface value in Flow360 is calculated as:
  ```
  Q_default = RefMach² / (all wall's bounding box length)²
  ```

## Visualization Tips

### Boundary Layer Separation
* Use the streamwise component of `CfVec` to identify separated regions
* Set a visualization scale with 3 levels (e.g., -1e-6, 0, 1e-6) to easily distinguish between attached (positive values) and separated (negative values) flow regions

### Surface Flow Patterns
* Use surface streamlines with `CfVec` components instead of velocity
* This shows recirculation patterns on the surface

### Vortex Visualization
* Use `qcriterion` isosurfaces to identify vortices
* For airplane simulations: recommended isosurface value is approximately Mach²/WingSpan²
* For rotor-dominated flows: recommended isosurface value is approximately TipMach²/RotorDiameter²
* Larger isosurface values show only stronger vortices
* Smaller values show more flow features but may clutter visualization

### Boundary Layer Visualization
* Use `Cpt` (total pressure coefficient) to visualize boundary layer development
* Lower values (typically shown in blue) highlight boundary layer regions

## BET Visualization
When using Blade Element Theory (BET) models, the volumeOutput can include additional `betMetrics` that provide visualization of:

* Blade loading distributions
* Inducted velocities
* Local angle of attack
* Other BET-specific quantities

These metrics are useful for analyzing rotor and propeller performance.

## History Files

Flow360 generates various history files that record time-series data during simulations:

### Actuator Disk Output
* Records thrust, torque, and power for actuator disk models
* Useful for tracking propulsion system performance

### BET Loading Output
* Records sectional forces and moments for blade elements
* Can be used to analyze blade loading distributions

### Aeroacoustic Output
* Records acoustic data at observer locations
* Used for noise prediction and analysis

### Heat Transfer
* Records heat flux and temperature information
* Important for thermal analysis applications
