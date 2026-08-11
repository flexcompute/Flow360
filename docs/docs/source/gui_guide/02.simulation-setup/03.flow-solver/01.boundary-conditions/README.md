# Boundary Conditions

*This section covers the various boundary conditions available in Flow360.*

Boundary conditions specify how the fluid behaves at the boundaries of the simulation domain. Boundaries conditions define the physical and mathematical conditions at the enclosed faces of the computational domain. Flow360 provides a comprehensive set of boundary conditions to support various simulation scenarios.

## Overview of Boundary Conditions

| *Option* | *Description* | *Key Parameters* |
|------------|----------------------------------|----------|
| **[Wall](./01.wall.md)** | Solid surface where fluid cannot penetrate | Wall Motion, Thermal Condition, Wall Function, Roughness Height |
| **[Freestream](./02.freestream.md)** | Far-field condition with uniform flow | Mach Number, Angle of Attack, Sideslip Angle |
| **[Inflow](./03.inflow.md)** | Boundary where flow enters the domain | Velocity Profile, Total Pressure, Flow Direction |
| **[Outflow](./04.outflow.md)** | Boundary where flow exits the domain | Static Pressure, Mass Flow Rate, Backflow Treatment |
| **[Periodic](./05.periodic.md)** | Repeating boundary condition for cyclic geometries | Rotation Angle, Translation Vector |
| **[Symmetry](./06.symmetry.md)** | Boundary that enforces symmetry of the solution | Symmetry Plane |
| **[Slip Wall](./07.slip-wall.md)** | Wall with zero normal velocity but allows tangential flow | - |
| **[Porous Jump](./08.porous-jump.md)** | Thin porous interface imposing a pressure drop | Darcy Coefficients, Forchheimer Coefficient, Thickness |

Additionally, [Turbulence Quantities](./09.turbulence-quantities.md) are used to specify turbulence parameters at all types of boundaries, such as Turbulence Intensity, Eddy Viscosity Ratio, Length Scale, etc.

Click on each boundary condition type to see detailed documentation including available parameters, descriptions, usage tips, and example configurations.

```{toctree}
:hidden:
:maxdepth: 3
./01.wall.md
./02.freestream.md
./03.inflow.md
./04.outflow.md
./05.periodic.md
./06.symmetry.md
./07.slip-wall.md
./08.porous-jump.md
./09.turbulence-quantities.md
``` 