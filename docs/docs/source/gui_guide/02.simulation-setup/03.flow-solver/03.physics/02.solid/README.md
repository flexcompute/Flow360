# Solid model

*The Solid model is used for setting up conjugate heat transfer volume models that contain all the common fields every heat transfer zone should have. This model is essential for simulating heat transfer in solid materials within the Flow360 simulation environment.*

## Major Components

1.  **[Heat equation solver](./01.heat-equation-solver.md)**: Controls settings for the heat equation solver, including tolerances and iteration limits.
2.  **[Material](./02.material.md)**: Defines material properties of the solid, such as thermal conductivity, density, and specific heat capacity.
3.  **Volume heat source**: Specifies a heat source per unit volume within the solid material.
4.  **[Initial condition](./03.initial-condition.md)**: Sets initial temperature field for the heat equation.
5.  **Assigned zones**: A list of volume zones where the heat transfer equations will be solved.

```{important}
Solid (conjugate heat transfer) zones must be meshed with tetrahedra — use the **Tetrahedra** element type when defining custom solid zones.
```

```{toctree}
:hidden:
:maxdepth: 3
./01.heat-equation-solver.md
./02.material.md
./03.initial-condition.md
```
