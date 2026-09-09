.. _user_defined_dynamic_user_guide:

User Defined Dynamics
=====================

User Defined Dynamics (UDD) enables you to implement custom control laws and coupled physics within your CFD simulations. It creates a feedback loop where flow solution quantities influence simulation parameters in real-time, allowing you to simulate complex coupled phenomena such as active control systems, aeroelastic interactions, and adaptive boundary conditions.

How It Works
------------

User Defined Dynamics operates as a computational layer that runs alongside your CFD simulation. At each time step, the system executes the following sequence:

1. **Reads from the flow solution**: Input variables (specified via ``input_vars``) extract quantities from the flow field such as forces, moments, pressure coefficients, or BET disk variables. These values are computed from the current flow solution and made available to your dynamics system. For input variables that have source surfaces (such as ``CL`` or ``forceX``), you can specify ``input_boundary_patches`` to choose the surfaces from which the values will be computed.

2. **Updates state variables**: State variables evolve according to mathematical expressions you define in ``update_law``, with initial values specified in ``state_vars_initial_value``. The update law expressions are evaluated at each pseudo-step and compute new values for the state variables based on the current state, input variables, constants (defined via ``constants``), and simulation variables (like ``pseudoStep``, ``timeStepSize``, ``t``). 

   **Important**: The ``update_law`` is used exclusively to update state variables. It does not directly control input or output variables. State variables serve as internal memory for your dynamics system, enabling you to implement:
   
   * Control laws (e.g., PI controllers that maintain target lift coefficients)
   * Structural dynamics (e.g., spring-mass-damper systems for aeroelastic coupling)
   * Accumulators (e.g., integral terms in control systems)
   * Any custom mathematical relationships

3. **Computes output variables**: Output variables (specified via ``output_vars``) compute values from the updated state variables. These output values directly modify simulation parameters, creating the feedback loop—changes to the flow solution affect the dynamics, which in turn affect the flow solution.

   **Entity association**: Some output variables are associated with specific entities and require you to specify ``output_target``. For example, output variables like ``theta`` (rotation angle), ``omega`` (angular velocity), and ``omegaDot`` (angular acceleration) control rotation of a volume zone and must have ``output_target`` set to the target entity (e.g., a rotating volume zone such as a ``Cylinder``). Other output variables, such as ``alphaAngle`` and ``betaAngle``, apply globally and do not require an ``output_target``.

The workflow is defined using the :class:`~flow360.UserDefinedDynamic` class, which encapsulates all these parameters into a single configuration.

.. note::
   All expressions and variables entered in User Defined Dynamics must be C-syntax compatible. For details on the syntax requirements and supported operations, see :ref:`Legacy User Expressions <expressions_legacy_user_guide>`.

Examples
--------

The example library contains end to end examples of how to use User Defined Dynamics for various applications:

* :doc:`UDD Alpha Controller </python_api/example_library/notebooks/udd_alpha_controller>`: Implements a PI (Proportional-Integral) controller that automatically adjusts the angle of attack to maintain a target lift coefficient. This example demonstrates reading flow solution quantities (lift coefficient), implementing control logic with state variables, using conditional expressions based on ``pseudoStep``, and associating input variables with specific boundary patches.

* :doc:`Flat Plate with Structural Aerodynamic Load </python_api/example_library/notebooks/flat_plate_spring>`: Demonstrates coupling structural dynamics with the flow solution. A flat plate rotates in response to aerodynamic moments, with the rotation governed by a torsional spring-mass-damper system. This example shows how to implement multiple state variables, control rotation of a volume zone through User Defined Dynamics, and use output variables (``omegaDot``, ``omega``, ``theta``) that control mesh rotation.
