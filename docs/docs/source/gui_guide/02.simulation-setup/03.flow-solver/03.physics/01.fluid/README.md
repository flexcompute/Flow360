# Fluid model

*The Fluid model represents the primary medium for CFD simulations in Flow360. It integrates several components that together govern the fluid dynamics behavior, including the Navier-Stokes solver, turbulence modeling, transition effects, and initial conditions. The Fluid model is applied to volume entities within your simulation domain.*

```{note}
The freestream state (density, temperature, and velocity) is specified in the **[Operating Condition](../../../01.flow-conditions/README.md)** section, not in the Fluid model. The Fluid model's **[Material](./07.material.md)** section describes the substance itself (viscosity model, thermodynamics, and Prandtl numbers for a gas, or density and viscosity for a liquid). This separation allows the same Fluid model configuration to be used with different conditions.
```

## Major Components

The Fluid model consists of seven primary components, each documented in detail in its own section:

1. [**Navier-Stokes Solver**](./01.navier-stokes-solver.md): Controls the core flow equations that govern momentum, continuity, and energy in the fluid. This component determines how the simulation resolves velocity, pressure, and density fields.

2. [**Turbulence Model**](./02.turbulence-model.md): Handles the modeling of turbulent flow structures through various approaches such as Spalart-Allmaras or k-Omega SST. This significantly impacts flow separation prediction and overall solution accuracy.

3. [**Transition Model**](./03.transition-model.md): Determines how and when flow transitions from laminar to turbulent within the simulation, which is critical for correctly predicting aerodynamic performance, especially at moderate Reynolds numbers.

4. [**Initial Condition**](./04.initial-condition.md): Defines the starting flow state for the simulation, which can significantly impact convergence rates and stability, especially for complex flows.

5. [**Stopping Criteria**](./05.stopping-criteria.md): Allows automatic termination of the simulation when monitored output fields (forces, probe values, or surface probe data) reach specified tolerance thresholds, providing efficient convergence control.

6. [**Gravity**](./06.gravity.md): Applies a gravitational body force to the fluid momentum and energy equations, enabling simulation of buoyancy-driven flows. Disabled by default.

7. [**Material**](./07.material.md): Defines the physical properties of the simulated fluid: the viscosity model, thermally perfect gas species, and Prandtl numbers for a gas, or the density and viscosity for a liquid.

```{seealso}
Knowledge base: {doc}`/knowledge_base/Simulation/navierStokesSolver/navierStokesSolver` and {doc}`/knowledge_base/Simulation/turbulenceModelSolver/turbulenceModelSolver`.

For how to define custom stopping criteria, see the {doc}`/user_guide/RunControl/RunControl` user guide.
```

```{toctree}
:hidden:
:maxdepth: 3
./01.navier-stokes-solver.md
./02.turbulence-model.md
./03.transition-model.md
./04.initial-condition.md
./05.stopping-criteria.md
./06.gravity.md
./07.material.md
```
