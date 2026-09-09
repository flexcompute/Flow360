# Physics

*This section covers the physical models that simulate the 3D flow behavior in Flow360.*

## Available Models

| *Model* | *Description* | *Key Parameters* |
|------------|----------------------------------|----------|
| **[Fluid](./01.fluid/README.md)** | Fluid model | [Navier-Stokes solver](./01.fluid/01.navier-stokes-solver.md), [Turbulence model](./01.fluid/02.turbulence-model.md), [Transition model](./01.fluid/03.transition-model.md), [Initial Condition](./01.fluid/04.initial-condition.md), [Gravity](./01.fluid/06.gravity.md) |
| **[Solid](./02.solid/README.md)** | Solid body model for conjugate heat transfer | [Heat equation settings](02.solid/01.heat-equation-solver.md), [Material](./02.solid/02.material.md), [Initial condition](./02.solid/03.initial-condition.md) |
| **[Rotation](./03.rotation.md)** | Handling of rotating components | Rotation type (MRF/SRF/Physical), Angular velocity |
| **[BET disk](./04.bet-disk.md)** | Blade Element Theory for propeller/rotor modeling | Polars, RPM |
| **[Actuator disk](./05.actuator-disk.md)** | Simplified model for propellers and rotors | Thrust coefficient, Swirl distribution |
| **[Porous medium](./06.porous-medium.md)** | Model for flow through porous regions | Darcy coefficient, Forchheimer coefficient |

Click on each model to see detailed documentation including available parameters, descriptions, usage tips, and example configurations.

---

## Detailed Descriptions

### [Fluid](./01.fluid/README.md)

*Modelling of the fluid behaviour in the domain.*

**Subsections:**
- **[Navier-Stokes solver](./01.fluid/01.navier-stokes-solver.md)** - Core flow solver configuration
- **[Turbulence model](./01.fluid/02.turbulence-model.md)** - Advanced turbulence modeling options including RANS, LES, and hybrid approaches
- **[Transition model](./01.fluid/03.transition-model.md)** - Laminar-to-turbulent transition prediction for improved accuracy
- **[Initial condition](./01.fluid/04.initial-condition.md)** - Flow field initialization strategies and convergence acceleration
- **[Gravity](./01.fluid/06.gravity.md)** - Gravitational body force for buoyancy-driven flow simulations

### [Solid](./02.solid/README.md)

*Conjugate heat transfer modeling for solid materials enabling accurate thermal analysis of components in contact with fluid flow. Provides material property specification, heat equation solver configuration, and thermal boundary condition management for multi-physics simulations.*

**Subsections:**
- **[Heat equation solver](./02.solid/01.heat-equation-solver.md)** - Thermal conduction solver settings and material property configuration
- **[Material](02.solid/02.material.md)** - Material properties specification
- **[Initial condition](02.solid/03.initial-condition.md)** - The initial state of solid bodies

### [Rotation](./03.rotation.md)

*Advanced rotating component handling with multiple approaches for modeling rotating machinery, propellers, and turbines. Supports Moving Reference Frame (MRF), Sliding Reference Frame (SRF), and physical rotation methods with comprehensive angular velocity specification and interface treatment.*

**Key Features:**
- Multiple rotation types (MRF/SRF/Physical) for different applications
- Angular velocity specification and rotational axis definition
- Interface treatment between rotating and stationary regions
- Support for complex multi-rotor configurations

### [BET disk](./04.bet-disk.md)

*Blade Element Theory implementation for high-fidelity propeller and rotor modeling. Enables detailed aerodynamic analysis of rotating blades through sectional force calculations, polar data integration, and performance prediction for aerospace and marine applications.*

**Key Features:**
- Sectional aerodynamic force calculation using polar data
- RPM and blade geometry specification
- Performance prediction including thrust and power
- Support for complex blade geometries and operating conditions

### [Actuator disk](./05.actuator-disk.md)

*Simplified propeller and rotor modeling approach using momentum theory. Provides efficient representation of rotating components through thrust coefficient specification and swirl distribution modeling, ideal for preliminary design and optimization studies.*

**Key Features:**
- Thrust coefficient specification for performance modeling
- Swirl distribution control for realistic wake effects
- Efficient computation for design optimization
- Support for multiple actuator disk configurations

### [Porous medium](./06.porous-medium.md)

*Advanced modeling of flow through porous regions including filters, heat exchangers, and porous materials. Implements Darcy-Forchheimer theory with comprehensive coefficient specification for accurate pressure drop and flow distribution prediction.*


```{toctree}
:hidden:
:maxdepth: 3
./01.fluid/README.md
./02.solid/README.md
./03.rotation.md
./04.bet-disk.md
./05.actuator-disk.md
./06.porous-medium.md
```
