# Simulation Setup

*This section covers all aspects of setting up a simulation in Flow360, including flow conditions, meshing, solver configuration, and output settings.*

## Contents

| *Section* | *Description* |
|-------------|-----------------|
| [Flow Conditions](./01.flow-conditions/README.md) | Define freestream and operating conditions |
| [Mesh](./02.mesh/README.md) | Configure mesh generation, rotation regions, and refinements |
| [Flow Solver](./03.flow-solver/README.md) | Set solver numerics, boundary conditions, and physics models |
| [Output](./04.output/README.md) | Specify output fields, reference dimensions, and result formats |

---

## Detailed Subsections

- **Flow Conditions:**
  - [Operating condition](./01.flow-conditions/README.md): Define physical values of fluid properties

- **Mesh:**
  - [Farfield](./02.mesh/01.farfield.md): Outer boundaries and domain configuration
  - [Mesh parameters](./02.mesh/02.mesh-parameters.md): Control parameters for mesh generation
  - [Rotation zones](./02.mesh/03.rotation-zones.md): Specialized regions for rotating features
  - [Custom zones](./02.mesh/04.custom-zones.md): User-defined volume zones
  - [Volume mesh slices](./02.mesh/05.volume-mesh-slices.md): Slice outputs from the volume mesh
  - [Refinements](./02.mesh/06.refinements/README.md): Local mesh control for critical regions

- **Flow Solver:**
  - [Boundary conditions](./03.flow-solver/01.boundary-conditions/README.md): Define domain boundaries
  - [Time](./03.flow-solver/02.time.md): Time resolution controls
  - [Physics](./03.flow-solver/03.physics/README.md): Physical models in the simulation
  - [User-defined dynamics](./03.flow-solver/04.user-defined-dynamics.md): Custom dynamics definitions

- **Output:**
  - [Reference dimensions](./04.output/reference-dimensions.md): Dimensions for force coefficient calculations
  - [Output list](./04.output/02.outputs-list/README.md): All outputs to generate from the simulation
  - [Available Output Fields](./04.output/output-fields.md): Complete reference of flow variables by output type
  - [Scaling Values](./04.output/scaling-values.md): Reference values for nondimensionalization
  

```{toctree}
:hidden:
:maxdepth: 3
./01.flow-conditions/README.md
./02.mesh/README.md
./03.flow-solver/README.md
./04.output/README.md
```