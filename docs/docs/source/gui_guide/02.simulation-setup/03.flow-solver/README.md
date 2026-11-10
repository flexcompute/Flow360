# Flow solver 
A section for setting the solver numerics and domain boundary conditions.

## Contents

| *Option* | *Description* |
|------------|----------------------------------|
| **[Boundary conditions](./01.boundary-conditions/README.md)** | Boundary condition definitions |
| **[Time](./02.time.md)** | Controls for the time resolution in the simulation |
| **[Physics](./03.physics/README.md)** | A set of physical models that are found in the simulation |
| **[User-defined dynamics](./04.user-defined-dynamics.md)** | Definition of user-defined dynamics |

---

## Detailed Descriptions

### [Boundary conditions](./01.boundary-conditions/README.md)

*Comprehensive set of boundary conditions including walls, freestream, inflow/outflow, periodic, symmetry, and slip wall conditions. Defines how fluid behaves at domain boundaries with support for wall motion, thermal conditions, turbulence quantities, and various flow specifications.*

**Subsections:**
- **[Wall](./01.boundary-conditions/01.wall.md)** - Solid surface boundary conditions with wall motion, thermal conditions, and roughness
- **[Freestream](./01.boundary-conditions/02.freestream.md)** - Farfield conditions with uniform flow specifications
- **[Inflow](./01.boundary-conditions/03.inflow.md)** - Flow entry boundaries with velocity profiles and pressure conditions
- **[Outflow](./01.boundary-conditions/04.outflow.md)** - Flow exit boundaries with static pressure and mass flow control
- **[Periodic](./01.boundary-conditions/05.periodic.md)** - Repeating boundary conditions for cyclic geometries
- **[Symmetry](./01.boundary-conditions/06.symmetry.md)** - Symmetry plane enforcement for computational efficiency
- **[Slip Wall](./01.boundary-conditions/07.slip-wall.md)** - Zero normal velocity walls allowing tangential flow
- **[Turbulence Quantities](./01.boundary-conditions/09.turbulence-quantities.md)** - Turbulence parameter specifications for all boundary types

### [Time](./02.time.md)

*Advanced time stepping configuration for both steady and unsteady simulations. Controls simulation type (steady/unsteady), CFL management (Adaptive/Ramp), convergence parameters, physical time steps, pseudo-steps, and temporal accuracy. Includes detailed guidelines for time step selection based on vortex shedding frequencies and rotational motion.*

### [Physics](./03.physics/README.md)

*Complete physical modeling framework encompassing fluid dynamics (Navier-Stokes solver, turbulence models, transition models, initial conditions), heat transfer (conjugate heat transfer in solids), and special models (rotation, BET disk, actuator disk, porous media). Provides comprehensive parameter control for accurate 3D flow simulation.*

**Subsections:**
- **[Fluid](./03.physics/01.fluid/README.md)** - Core fluid dynamics including Navier-Stokes solver, turbulence models, transition models, and initial conditions
- **[Solid](./03.physics/02.solid/README.md)** - Conjugate heat transfer in solid materials with material properties and heat equation solver
- **[Rotation](./03.physics/03.rotation.md)** - Rotating component handling
- **[BET disk](./03.physics/04.bet-disk.md)** - Blade Element Theory for propeller/rotor modeling
- **[Actuator disk](./03.physics/05.actuator-disk.md)** - Simplified propeller and rotor modeling
- **[Porous medium](./03.physics/06.porous-medium.md)** - Flow through porous regions modeling

### [User-defined dynamics](./04.user-defined-dynamics.md)

*Advanced capability for implementing custom time-dependent motion of simulation components. Enables complex trajectories, rotations, and dynamic interactions between aerodynamic forces and component movement. Currently optimized for Python API implementation with enhanced debugging and flexibility for sophisticated dynamic scenarios.*


```{toctree}
:hidden:
:maxdepth: 3
./01.boundary-conditions/README.md
./02.time.md
./03.physics/README.md
./04.user-defined-dynamics.md
```