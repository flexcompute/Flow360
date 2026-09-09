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

## Zone-zone interfaces

When a project starts from an uploaded volume mesh with more than one zone, the boundary pairs
where two zones meet are detected automatically and coupled, so the flow passes from one zone
into the next. Interface faces are not boundaries of the simulation domain. They carry no
boundary condition, and leaving them coupled is the correct setup for a multi-zone mesh.

To measure the flow crossing an interface, assign a
[Surface Integral](../../04.output/02.outputs-list/20.surface-integral-output.md) output to it.
Outputs observe an interface without altering it, and require no change to the coupling.

A boundary condition assigned to an interface face **replaces** the coupling on that pair. This
is not part of regular case setup. It exists for the narrow case where a passage must be closed
off in a mesh that cannot be regenerated — examining a configuration with an inlet shut, for
instance. Where the geometry is known in advance, mesh the configuration you intend to simulate
instead.

An assignment that replaces the coupling is rejected unless:

- **Both faces of the pair are assigned.** An interface has one face in each of the two zones;
  assigning only one is an error.
- **Both zones are static fluid zones.** A rotating zone needs its interfaces for the
  sliding-interface interpolation, and a solid zone for conjugate heat transfer, so the
  interfaces of such zones cannot be reassigned.
- **The model is not [Periodic](./05.periodic.md).** An interface face is already paired with the
  coincident face in the neighbouring zone and cannot also be paired with a periodic image face.

One further limitation is **not** checked and must be respected by the setup: where two zones
share more than one interface patch, either all of them are assigned or none. Replacing the
coupling on a subset leaves a partial baffle across the shared boundary, which the solver does
not support.

To impose a pressure drop across an interface while keeping the zones coupled, use
[Porous Jump](./08.porous-jump.md), which is applied on the coupling rather than replacing it.

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