.. _user_defined_dynamic_user_guide:

User Defined Dynamics
=====================

User Defined Dynamics (UDD) enables you to implement custom control laws and coupled physics within your CFD simulations. It creates a feedback loop where flow solution quantities influence simulation parameters in real-time, allowing you to simulate complex coupled phenomena such as active control systems, aeroelastic interactions, and adaptive boundary conditions.

How It Works
------------

User Defined Dynamics operates as a computational layer that runs alongside your CFD simulation. At each time step, the system executes the following sequence:

1. **Reads from the flow solution**: Input variables (specified via ``input_vars``) extract quantities from the flow field such as forces, moments, pressure coefficients, or BET disk variables. These values are computed from the current flow solution and made available to your dynamics system. For input variables that have source surfaces (such as ``CL`` or ``forceX``), you can specify ``input_boundary_patches`` to choose the surfaces from which the values will be computed.

2. **Updates state variables**: State variables evolve according to mathematical expressions you define in ``update_law``, with initial values specified in ``state_vars_initial_value``. The update law expressions are evaluated at each pseudo-step and compute new values for the state variables based on the current state, input variables, constants (defined via ``constants``), and simulation variables (like ``pseudoStep``, ``timeStepSize``, ``physicalTime``).

   **Important**: The ``update_law`` is used exclusively to update state variables. It does not directly control input or output variables. State variables serve as internal memory for your dynamics system, enabling you to implement:
   
   * Control laws (e.g., PI controllers that maintain target lift coefficients)
   * Structural dynamics (e.g., spring-mass-damper systems for aeroelastic coupling)
   * Accumulators (e.g., integral terms in control systems)
   * Any custom mathematical relationships

3. **Computes output variables**: Output variables (specified via ``output_vars``) compute values from the updated state variables. These output values directly modify simulation parameters, creating the feedback loop—changes to the flow solution affect the dynamics, which in turn affect the flow solution.

   **Entity association**: Some output variables are associated with specific entities and require you to specify ``output_target``. For example, output variables like ``theta`` (rotation angle), ``omega`` (angular velocity), and ``omegaDot`` (angular acceleration) control rotation of a volume zone and must have ``output_target`` set to the target entity (e.g., a rotating volume zone such as a ``Cylinder``). Other output variables, such as ``alphaAngle`` and ``betaAngle``, apply globally and do not require an ``output_target``.

The workflow is defined using the :class:`~flow360.UserDefinedDynamic` class, which encapsulates all these parameters into a single configuration.

.. note::
   All expressions and variables entered in User Defined Dynamics must be C-syntax compatible. For details on the syntax requirements and supported operations, see :ref:`Legacy User Expressions <expressions_legacy_user_guide>`. The expression fields take strings only, so :py:class:`~flow360.UserVariable` objects cannot be used in them.

.. _udd_boundary_to_boundary:

Driving one set of boundaries from another
------------------------------------------

A device such as a fan, an air handler or a scrubber takes flow in through one set of faces and returns it through another. A dynamic models it by measuring the intake side and driving the supply side.

Inputs are summed over the input boundaries. An input named ``target_<name>`` reads the same quantity summed over the output targets instead, so one dynamic can compare both sides, for example ``massFlowRate`` on the intake against ``target_massFlowRate`` on the supply. The output target can be a list of surfaces, and every output is then written to each of them.

Each output can be written by one dynamic only. A case where two dynamics write the same output on the same boundary is rejected, because the dynamics run in order and the second would silently overwrite the first.

To set the total pressure or total temperature of a supply face, write ``totalPressureRatio`` (total pressure over the freestream static pressure) or ``totalTemperatureRatio`` (total temperature over the freestream temperature) on an :py:class:`~flow360.Inflow`:

* ``totalPressureRatio`` works on an inflow with a :py:class:`~flow360.TotalPressure` spec whose value is a constant, or on an inflow with a :py:class:`~flow360.MassFlowRate` spec. On a mass flow rate inflow, the dynamic replaces the built-in controller that holds the mass flow at its target, so the target is no longer enforced.
* ``totalTemperatureRatio`` works on an inflow with a :py:class:`~flow360.TotalPressure` or :py:class:`~flow360.MassFlowRate` spec whose total temperature is a constant.

The solver rejects either output on any other boundary, including a :py:class:`~flow360.Supersonic` inflow, or where the total pressure or total temperature is an expression.

Where to set it: :py:attr:`~flow360.UserDefinedDynamic.input_boundary_patches`, :py:attr:`~flow360.UserDefinedDynamic.output_target`.

.. _udd_species_composition:

Controlling the composition of an inflow
----------------------------------------

In a closed room with a crew breathing or a CO\ :sub:`2` scrubber running, what leaves the room changes what the device puts back. With a transported gas mixture (see :ref:`formulation_gas_mixtures`), a dynamic reads the mass flow of each species leaving through one boundary and sets the composition entering through an inflow.

- **Reading**: ``massFlowRate_<species>`` is the mass flow of one species through the input boundaries. It is available, like ``massFlowRate``, on mass flow inflows and outflows, pressure outflows, total pressure inflows and porous bleed walls. Inflows report a positive mass flow.
- **Writing**: ``massFraction_<species>`` sets the mass fraction of one species on an :class:`~flow360.Inflow` with a :class:`~flow360.TotalPressure` specification. Write any species except the last one declared on the gas, whose mass fraction is the remainder of the others.

The solver keeps the written composition physical. Values outside zero to one are clipped, and a set that sums above one is rescaled, with a warning in the case log the first time it happens on a boundary. A value that is not finite stops the case with an error that names the boundary.

.. note::
   A room that starts at rest has zero mass flow on every boundary at the first step. A law that divides by a mass flow needs a guard, for example keeping its previous state while the flow is zero.

The dynamic below models a scrubber that removes 90% of the CO\ :sub:`2` reaching its return face and sends the rest back through its supply face:

.. code-block:: python

   scrubber = fl.UserDefinedDynamic(
       name="scrubber",
       input_vars=["massFlowRate_CO2", "target_massFlowRate"],
       constants={"efficiency": 0.9},
       state_vars_initial_value=["0.0"],
       update_law=[
           "(target_massFlowRate > 0.0) ? (1.0 - efficiency) * massFlowRate_CO2 / target_massFlowRate : state[0]"
       ],
       output_vars={"massFraction_CO2": "state[0]"},
       input_boundary_patches=[return_face],
       output_target=[supply_face],
   )

Where to set it: :py:attr:`~flow360.UserDefinedDynamic.input_vars`, :py:attr:`~flow360.UserDefinedDynamic.output_vars`, :py:attr:`~flow360.Inflow.spec`.

Examples
--------

The example library contains end to end examples of how to use User Defined Dynamics for various applications:

* :doc:`UDD Alpha Controller </python_api/example_library/notebooks/udd_alpha_controller>`: Implements a PI (Proportional-Integral) controller that automatically adjusts the angle of attack to maintain a target lift coefficient. This example demonstrates reading flow solution quantities (lift coefficient), implementing control logic with state variables, using conditional expressions based on ``pseudoStep``, and associating input variables with specific boundary patches.

* :doc:`Flat Plate with Structural Aerodynamic Load </python_api/example_library/notebooks/flat_plate_spring>`: Demonstrates coupling structural dynamics with the flow solution. A flat plate rotates in response to aerodynamic moments, with the rotation governed by a torsional spring-mass-damper system. This example shows how to implement multiple state variables, control rotation of a volume zone through User Defined Dynamics, and use output variables (``omegaDot``, ``omega``, ``theta``) that control mesh rotation.
