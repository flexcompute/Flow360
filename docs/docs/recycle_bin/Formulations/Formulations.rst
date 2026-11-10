.. _formulations:

Formulations
============
Flow360 is a **Navier-Stokes** solver that solves flow equations and offers multiple models to describe the physical domain and represent phenomena like turbulence or heat transfer.
Models can be divided into:

* **2D models** - the *boundary conditions* that describe the behaviour at the surfaces of the domain.
* **3D models** - are used to define the behaviour of the volumetric regions of the domain.

Additionally, this section will describe:

* **Turbulence quantities** - the values that can be set on certain **2D models** to define the boundary conditions for the turbulence model solver.

.. _surf_models:

2D models (boundary conditions)
-------------------------------
Boundary conditions define the physical behavior at the surfaces that bound your computational domain. They are essential for accurately representing the interaction between the flow and its surroundings, such as solid walls, far-field regions, or inflow and outflow boundaries. Choosing the correct boundary conditions ensures that your simulation captures the real-world physics of your problem, whether it's air flowing over a wing, fluid entering through an inlet, or periodic flow patterns.

Flow360 provides several boundary condition types to handle different physical scenarios.

Symbol explanation:

* :math:`\mathbf{n}` - a unit vector normal to the boundary,
* :math:`\mathbf{t}_i` - orthogonal unit vectors tangent to the boundary,
* :math:`\mathbf{v} = [u, v, w]^T` - velocity vector,
* :math:`\rho` - fluid density,
* :math:`p` - pressure.

A state (:math:`\mathbf{q}`) is a set of primitive variables defining the flow state for a certain node:

.. math::

   \mathbf{q} = \begin{bmatrix} \rho \\ u \\ v \\ w \\ p \end{bmatrix} = \begin{bmatrix} \rho \\ \mathbf{v} \\ p \end{bmatrix}

In a volume node the state of a node is solved for iteratively based on the current state of the node :math:`\mathbf{q}_{node}` and the states of its neighbouring nodes :math:`[\mathbf{q}_{neighbour_0}, ...]`. Boundary conditions are achieved by setting certain state variables of the boundary node and introducing an outside state :math:`\mathbf{q}_{out}` that represents state variables "outside" the computational domain.

.. _wall_formulations:

Wall
^^^^
The most common boundary condition defined by the velocity fixed at the boundary nodes:

.. math::

   \mathbf{v}_{node} = \mathbf{v}_{def} \cdot \mathbf{t}_0 + \mathbf{v}_{def} \cdot \mathbf{t}_1

And the other state variables solved, with outside state:

.. math::

   \mathbf{q}_{out} = \mathbf{q}_{node}
 
With :math:`\mathbf{v}_{def}` being a wall velocity defined by the user, usually :math:`\mathbf{v}_{def} = 0`.

The presence of the wall influences the volume cells through the *wall function* that is used to resolve the boundary layer formation with grids that are not adapted for direct resolution.
The *wall function* depends on the distance to the wall.

The **Wall** boundary condition can also be used to define the roughness height of the wall. Roughness is specified through the **Roughness height** parameter, which is equal to the equivalent sand grain roughness height.

.. admonition:: Tip
   :class: tip
   
   A conversion from real, geometrical roughness to an equivalent sand grain roughness is possible. See `this paper <https://ijmem.avestia.com/2012/PDF/008.pdf>`__ for more details.

Surface roughness creates a downward shift in the log layer velocity profile and increases shear stress at the wall compared to a smooth case. 
That shift is achieved through modifying the turbulent quantities near the wall.


.. _freestream:

Freestream
^^^^^^^^^^
Used to define a boundary with a constant exterior state.
It is realized by setting a fixed neighbouring state to the outside of the boundary faces. The node state is solved for.

.. math::

   \mathbf{q}_{out} = \begin{bmatrix} \rho_{\infty} \\ \mathbf{v}_{\infty} \\ p_{\infty} \end{bmatrix}

Where :math:`\infty` subscript means the freestream values defined by the user in operating condition.


.. admonition:: Important
   :class: important
   
   The velocity, density and temperature of the freestream flow are inferred from the *Operating condition*.
   The velocity can be overwritten per boundary condition in the Freestream interface.

.. _inflow:

Inflow
^^^^^^
Specifies flow entering the computational domain, allowing you to define inlet conditions for internal flow simulations or when inlet conditions differ from the global freestream.
How much fluid enters the domain can be controlled through the following specifications:

* **Total pressure** - the stagnation pressure at the inflow.
* **Mass flow rate** - the mass flow rate coming through the inflow. It is effectively realized by a controller that adjusts pressure at the boundary to achieve the target rate.
* **Supersonic** - when the incoming flow is supersonic, requires specifying both total and static pressure.

The Inflow boundary condition is enforced by using a specific outside state :math:`\mathbf{q}_{out}`. 
The formulation of that state ensures the variable that is specified to be retained at the boundary.

For **subsonic flows**, the full flow state is not specified at the inflow boundary. When total temperature, velocity direction, and total pressure are specified, the static pressure at the inflow is only determined after convergence of the solution. This occurs because in subsonic flow, one of the characteristics travels upstream, so information cannot be fully determined at the upstream boundary. The solver adjusts the boundary state iteratively based on the interior solution until convergence is achieved.

For the **supersonic flows**, both total and static pressure are directly specified, fully determining the flow state at the inlet. This is necessary because in supersonic flows, information cannot propagate upstream, so the complete flow state must be prescribed at the boundary.


Outflow
^^^^^^^
Defines where flow exits the computational domain, allowing you to specify conditions at outlet boundaries for internal flow simulations or when exit flow conditions are needed.

The outflow can be defined using following quantities:

* **Mach** - a target Mach number at the Outflow.
* **Pressure** - static pressure at the Outflow boundary.
* **Mass flow rate** - a target mass flow through the outflow. It is effectively realized by a controller that adjusts pressure at the boundary to achieve the target rate.

The Outflow boundary condition is enforced in an analogical way to the Inflow condition - by assigning a specific local :math:`\mathbf{q}_{out}`.


Periodic
^^^^^^^^
Links two separate boundaries so that flow exiting one boundary enters the other, enabling simulation of infinitely repeating geometry with a reduced computational domain.
A *translational* and *rotational* periodicity is available.

Assuming a and b are coupled nodes on a surface pair that forms the periodic boundary condition.

The state :math:`\mathbf{q}_a` is computed according to its own previous state and the states of its neighbourhood, which is a neighbourhood of both node a and b :math:`[\mathbf{q}_{a_{neighbour_0}}, ..., \mathbf{q}_{b_{neighbour_0}}, ...]` and vice versa.

.. admonition:: Important
   :class: important

   The mesh on the periodic boundaries has to be conformal.

Symmetry
^^^^^^^^
Used to reduce computational domain size by utilizing geometric and flow symmetries. They allow you to simulate only half of the full domain when the flow physics and geometry exhibit symmetrical properties.

Is computed by setting the outside state's velocity to the cells velocity mirrored by the boundary.

.. math::

   \mathbf{q}_{out} = \begin{bmatrix} \rho_{node} \\ \mathbf{v}_{node} - 2 \cdot \mathbf{v}_{node} \cdot \mathbf{n} \\ p_{node} \end{bmatrix}

And ensuring the gradients of primitive variables in directions tangential to the symmetry are zero.

.. math::

   \frac{d \rho}{d \mathbf{t}_i} = \nabla (\mathbf{v} \cdot \mathbf{t}_i) = \frac{d \rho}{d \mathbf{t}_i} = 0


Slip wall
^^^^^^^^^
A boundary condition used to define walls without any resistance from the surface in the tangential directions.

Is computed by setting the outside state's velocity to the cells velocity mirrored by the boundary.

.. math::

   \mathbf{q}_{out} = \begin{bmatrix} \rho_{node} \\ \mathbf{v}_{node} - 2 \cdot \mathbf{v}_{node} \cdot \mathbf{n} \\ p_{node} \end{bmatrix}

Which results in the following properties:

* no velocity normal to the Wall
* no wall shear stress → no tangential force exerted on the wall


Interface
^^^^^^^^^
.. admonition:: Crucial
   :class: danger
   
   The **interfaces** between volume zones are detected automatically. It is NOT necessary to set boundary conditions on surfaces which are supposed to become interfaces.

Interfaces are boundaries that have their counterparts in the neighbouring volume zones. Depending on the type of interface they have to be of certain type and share certain information.

* **fluid-fluid** interfaces:
   * **type**: non-conformal
   * **data transfer**: node interpolation
   * **variables**: all

* **fluid-solid** interfaces:
   * **type**: conformal
   * **data transfer**: direct coupling
   * **variables**: temperature


.. admonition:: Important
   :class: important

   A single surface should describe the interface between **only** two volume zones, otherwise the automatic detection will not work. 

.. _vol_models:

3D models 
---------

.. _fluid:

Fluid
^^^^^
Represents the primary medium for CFD simulations in Flow360. It integrates several components that together govern the fluid dynamics behavior, including the Navier-Stokes solver, turbulence modeling, transition effects, and initial conditions. The Fluid model is applied to volume entities within your simulation domain.

For each node in the computational mesh the fluid model is responsible for two solvers, which are solving their corresponding equations. 

1. **Navier-Stokes solver** - Responsible for solving the basic flow equations for primitive variables.

   * *momentum* - solves for :math:`\mathbf{v}`

   * *continuity* - solves for :math:`\rho`

   * *energy* - solves for :math:`p`


2. **Turbulence model solver** - Solving equations for providing the turbulent viscosity to the **Navier-Stokes solver**:

   *  **Spalart-Allmaras** model solves directly for *modified turbulent viscosity* (:math:`\hat{\nu}`):

   * **k-ω SST** solves for *kinetic turbulent energy* (:math:`k`) and *specific dissipation rate* (:math:`\omega`) to get the turbulent viscosity.

3. **Transition model solver** - the transition model solver is the AFT (Amplification Factor Transport) model. 
   The AFT model is based on linear stability theory, which aims to track the growth of pressure/velocity instabilities in the boundary layer. This approach more directly aligns with the physics involved with the transition process as compared to correlation based models. 

   Solves two additional equations: 

   * *intermittency*
   * *amplification factor*

   The implementation of the model is presented in more detail `here <https://turbmodels.larc.nasa.gov/aft_transition_3eqn.html>`__.

4. **Aeroacoustic solver** - provides the ability to estimate noise spectrum and level due to aerodynamics using an F1A-based Ffowcs Williams-Hawkings (FWH) aeroacoustics solver. This allows for the prediction of the acoustics pressure time history from acoustic sources that are primarily derived due to aerodynamical pressure fluctuations on solid surfaces. Flow360's aeroacoustics solver is source time-based, which results in higher accuracy and lower execution time as compared to observer time-based formulations.

   The FWH equations model is an inhomogeneous wave equation that can be derived from the Navier-Stokes equations by Lighthill's analogy. It contains many source terms that can contribute to the production of noise: monopole, dipole, and quadrupole. The first two terms are computed by taking integrals over a solid surface and the last term is a volumetric integral.
   In Flow360's aeroacoustic solver, we don't consider the contribution from quadrupole terms as it greatly increases the computational cost of the aeroacoustics solver. We only use the first two terms to estimate the amplitude of the acoustics signal at observer locations and our testing has shown that it is sufficient to accurately predict the sound pressure profile for a majority of aeroacoustics applications.

   Flow360's aeroacoustics solver supports a single SymmetryPlane boundary to mirror the acoustic signals for all no-slip wall boundaries.

.. admonition:: Important
   :class: important

   To enable an Aeroacoustic Solver it is required for a solution to be **Unsteady** and **Aeroacoustic Output** has to be added to the :ref:`Outputs <simulation_params_concept>` field.



.. _solid:

Solid 
^^^^^
Is used for setting up conjugate heat transfer volume models that contain all the common fields every heat transfer zone should have. This model is essential for simulating heat transfer in solid materials within the Flow360 simulation environment.

Solid is responsible for the **Heat model solver** that solves the *energy* equation to get the temperature of the solid.

Rotation
^^^^^^^^
Is used to described a volume rotating in reference to the parent reference frame.
Rotation is achieved in two ways: 

* through the MRF (moving reference frame) model,
* through a rotating mesh with sliding interfaces (unavailable in steady simulations).

Both ways are enforced by adding an appropriate source term to the momentum equation. The rotating mesh also actually rotates in relation to the parent volume.

.. _bet_form:

BET disk
^^^^^^^^
A reduced order model for representing propellers. In unsteady simulations it becomes a BET line model. 

It is defined using propeller geometrical data including the sections of the propeller, their CL and CD polars as well as a number of blades or the rotational speed.

Based on the parameters, an appropriate source term is added to the momentum equation within the volume the model is applied.

Actuator disk
^^^^^^^^^^^^^
A very simplified model of a propeller that represents it by adding a certain amount of momentum distributed over a disc-shaped region.

Based on the parameters, an appropriate source term is added to the momentum equation within the volume the model is applied.

.. _porous_form:

Porous medium
^^^^^^^^^^^^^
Used to simplify media that enforce a pressure resistance on the fluid but are not completely solid (eg. radiators or forrests).

The porous medium is realized by adding an appropriate source term to the transport equation. The resistance is modelled with the use of two coefficients:

* Darcy coefficient - responsible for the pressure drop due to the viscous effect.
* Forchheimer coefficient - responsible for the pressure drop due to the inertial effects.

Both coefficients are defined for each principal axis of the body they are defined on. 
It is also possible to specify a uniform volumetric heat source in the Porous medium.

Based on the parameters, an appropriate source term is added to the momentum equation within the volume the model is applied.
When the heat source is enabled, the source term is also added to the energy equation.

.. _turbulence_quantities:

Turbulence quantities
---------------------

Turbulence quantities can be specified for certain boundary conditions (:ref:`Inflow <inflow>`, :ref:`Freestream <freestream>`). They are used to define the values for turbulence variables:

* for the *Spalart-Allmaras* model:
   * nuHat (:math:`\hat{\nu}`) - modified turbulent viscosity
* for the *k-ω SST* model:
   * :math:`k` - turbulent kinetic energy
   * :math:`\omega` - specific dissipation rate

Turbulence quantities can be defined using the following:

* Turbulent Viscosity Ratio - The ratio between turbulent viscosity and laminar viscosity.
* Modified Viscosity Ratio - The ratio between modified turbulent viscosity (nuHat) and freestream laminar viscosity.
* Modified Viscosity (:math:`\hat{\nu}`) - The absolute value of the modified turbulent viscosity (nuHat).
* Turbulent Intensity (:math:`I`) - Relative turbulence level (as a decimal, not percentage).
   It is used to calculate the turbulent kinetic energy:

   .. math:: k = (U_{ref} \times I)^2

* Turbulent Length Scale - Characteristic size of turbulent eddies.
* Turbulent Kinetic Energy (:math:`k`) - Energy per unit mass in turbulent fluctuations. (direct specification)
* Specific Dissipation Rate (:math:`\omega`) - Rate at which turbulence energy converts to thermal energy. (direct specification)

Compatibility Matrix
^^^^^^^^^^^^^^^^^^^^

The following table shows which combinations of turbulence quantities are valid for different turbulence models:

.. list-table:: Turbulence Quantity Compatibility
   :header-rows: 1
   :stub-columns: 1
   :widths: 30 15 15 15 15 15 15 15
   :class: standard

   * -
     - **Turbulent Viscosity Ratio**
     - **Modified Viscosity Ratio**
     - **Modified Viscosity**
     - **Turbulent Kinetic Energy**
     - **Turbulent Intensity**
     - **Turbulent Length Scale**
     - **Specific Dissipation Rate**
   * - **Turbulent Viscosity Ratio**
     - SA+SST
     - x
     - x
     - SST
     - SST
     - SST
     - SST
   * - **Modified Viscosity Ratio**
     - x
     - SA
     - x
     - x
     - x
     - x
     - x
   * - **Modified Viscosity**
     - x
     - x
     - SA
     - x
     - x
     - x
     - x
   * - **Turbulent Kinetic Energy**
     - SST
     - x
     - x
     - SST
     - x
     - SST
     - SST
   * - **Turbulent Intensity**
     - SST
     - x
     - x
     - x
     - SST
     - SST
     - SST
   * - **Turbulent Length Scale**
     - SST
     - x
     - x
     - SST
     - SST
     - x
     - SST
   * - **Specific Dissipation Rate**
     - SST
     - x
     - x
     - SST
     - SST
     - SST
     - x

**Legend:**

* **SA** - Valid only for Spalart-Allmaras model
* **SST** - Valid only for :math:`k`-:math:`\omega` SST model
* **SA+SST** - Valid for both turbulence models
* x - Invalid combination

