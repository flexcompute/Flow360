.. _knowledge_base_navierStokesSolver:

.. currentmodule:: flow360

NavierStokesSolver
==================

:py:attr:`~NavierStokesSolver.absolute_tolerance`
-------------------------------------------------

The :py:attr:`~NavierStokesSolver.absolute_tolerance` is the primary convergence metric for steady cases. At least 5 orders of magnitude reduction is recommended for all residual values. The :py:attr:`~NavierStokesSolver.absolute_tolerance` can also be used for unsteady cases, but is less meaningful than the :py:attr:`~NavierStokesSolver.relative_tolerance`, as the initial residual values change between different physical steps.

.. _relativeTolerance: 

:py:attr:`~NavierStokesSolver.relative_tolerance`
-------------------------------------------------

The relative residual is defined as the ratio of the current pseudoStep's residual to the maximum residual present in the first 10 pseudoSteps within the current physicalStep. When running unsteady cases, the :py:attr:`~NavierStokesSolver.relative_tolerance` is typically set to 1e-2 or 1e-3. Once the nonlinear residuals drop by 2 or 3 orders of magnitude, the solver will continue to the next physicalStep. The :py:attr:`~NavierStokesSolver.relative_tolerance` is ignored for steady cases.

.. _kappaMUSCL:

:py:attr:`~NavierStokesSolver.kappa_MUSCL`
------------------------------------------

The default value of -1 leads to a second-order upwind scheme, which is the most stable. A value of 0.33 leads to a blended upwind/central scheme, which is recommended for low subsonic flows to reduce dissipation. Values greater than 0.33 are not recommended and a value of 1 leads to an unstable scheme.

.. _knowledge_base_orderOfAccuracy:

:py:attr:`~NavierStokesSolver.order_of_accuracy`
------------------------------------------------

The :py:attr:`~NavierStokesSolver.order_of_accuracy` determines whether the solver will use 1st or 2nd order spatial discretization. The 1st order solver is faster, cheaper and most importantly, it is more dissipative, making it less likely to diverge. However, such numerical dissipation may also significantly impact the accuracy of the solution.

When initializing the flow field for unsteady cases with rotating components, such as simulating a rotor enclosed in a sliding interface, the user may need to run the 1st-order solver for around 1 or 2 revolutions. Once the flow field has been initialized, the user can fork the first-order case and switch :py:attr:`~NavierStokesSolver.order_of_accuracy` from 1 to 2 for the child cases. 

While adjusting the :py:attr:`~NavierStokesSolver.order_of_accuracy` for the :class:`NavierStokesSolver`, the :ref:`TurbulenceModelSolver <knowledge_base_turbulenceModelSolver>` should also be adjusted.

The recommended :py:attr:`~SimulationParams.time_stepping` is slightly different for the 1st and 2nd order cases. For more details, see :ref:`Rotational Angle per Step <rot_angle_per_step>`, :ref:`maxPseudoSteps <knowledge_base_maxPseudoSteps>` and :ref:`CFL <knowledge_base_CFL>`

Limiters
--------

If the case is transonic or supersonic, the user should set :py:attr:`~NavierStokesSolver.limit_velocity` and :py:attr:`~NavierStokesSolver.limit_pressure_density` as :code:`TRUE` in the :class:`NavierStokesSolver` class.


:py:attr:`~NavierStokesSolver.linear_solver`
--------------------------------------------

:py:attr:`~NavierStokesSolver.linear_solver` controls the configuration for the linear solver. It includes information :py:attr:`~LinearSolver.max_iterations` which specifies the number of linear iteration performed in each pseudo-step. Typically, :py:attr:`~LinearSolver.max_iterations` is set to 25~35 for the NS solver. The user might need to increase it to 50-55 if the linear residual reduction ratio after linear solver is not enough. The default :py:attr:`~LinearSolver.max_iterations` for NS solver is 30.

Two linear solver types are available for the :class:`NavierStokesSolver`:

- :class:`LinearSolver` (default): standard iterative linear solver.
- :class:`KrylovLinearSolver`: Krylov iterative solver, available for **steady simulations only**.

.. _knowledge_base_krylovLinearSolver:

:class:`KrylovLinearSolver`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`KrylovLinearSolver` replaces the standard :class:`LinearSolver` with a Krylov method that offers improved convergence speed and reduced iteration counts for steady-state simulations. When the Krylov solver is selected, two additional features are automatically enabled:

1. **Enhanced adaptive CFL:** An adaptive CFL strategy based on nonlinear residual convergence is used, allowing the CFL to reach significantly higher values (up to ~10,000 for steady cases). The increased CFL makes each pseudo-step more effective at reducing residuals.
2. **Automatic evaluation frequency adjustment:** The Navier-Stokes equation and Jacobian evaluation frequencies are automatically set to 6, while the turbulence model equation and Jacobian update frequencies are set to 1. This results in 6 turbulence updates per Navier-Stokes update, helping both equation systems converge at a similar rate. The pseudo-step count reported by the solver corresponds to Navier-Stokes steps.

**Parameters:**

- :py:attr:`~KrylovLinearSolver.max_iterations`: Number of the Krylov iterations used per pseudo-step. Default: ``15``. Range: ``1``–``50``. Start with the default and reduce for better performance if the case converges well, or increase if convergence is slow.
- :py:attr:`~KrylovLinearSolver.max_preconditioner_iterations`: Number of preconditioner iterations applied during each Krylov iteration. Default: ``25``. Values between ``10``–``35`` are typically effective. Try increasing this value before increasing :py:attr:`~KrylovLinearSolver.max_iterations` if convergence is insufficient.
- :py:attr:`~KrylovLinearSolver.relative_tolerance`: Target relative convergence tolerance for the linear system. Default: ``0.05``. The adaptive CFL strategy uses this tolerance to decide when to increase or limit the CFL — setting it too low may prevent CFL from ramping up effectively. For low Mach number cases, values as low as ``1e-3`` may be needed for good convergence.

**Restrictions:**

.. warning::

   The :class:`KrylovLinearSolver` is subject to the following restrictions:

   - **Steady simulations only.** It cannot be used with :class:`~flow360.Unsteady` time stepping.
   - **Incompatible with velocity limiters.** :py:attr:`~NavierStokesSolver.limit_velocity` must be ``False``.
   - **Incompatible with pressure/density limiters.** :py:attr:`~NavierStokesSolver.limit_pressure_density` must be ``False``.
   - **Kappa MUSCL = 0.33 may cause convergence stalling.** If residuals stall at final convergence levels, try :py:attr:`~NavierStokesSolver.kappa_MUSCL` values of ``0`` or ``-0.33`` instead.

.. note::

   The low-dissipation scheme, MUSCL reconstruction, and low Mach preconditioner are all compatible with the Krylov solver.

:py:attr:`~NavierStokesSolver.line_search`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the :class:`KrylovLinearSolver`, :class:`LineSearch` can be configured via the :py:attr:`~NavierStokesSolver.line_search` field. The line search improves robustness by monitoring nonlinear residual growth between pseudo-steps and scaling back update steps that would cause non-physical or excessive residual increases.

The line search is coupled with the adaptive CFL strategy: the CFL is only increased when full steps are accepted, and is decreased when steps need to be scaled back. This coupling helps maintain stable convergence throughout the simulation.


**LineSearch parameters:**

- :py:attr:`~LineSearch.residual_growth_threshold`: Pseudotime nonlinear residual norm convergence ratio above which residual norm increase is allowed. Default: ``0.85``. Range: ``0.5``–``1.0``.
- :py:attr:`~LineSearch.max_residual_growth`: Hard cap on the residual norm ratio — never allow the residual norm to grow beyond this factor over a single pseudotime step. Default: ``1.1``. Must be ≥ ``1.0``.
- :py:attr:`~LineSearch.activation_step`: Pseudotime step threshold before the :py:attr:`~LineSearch.max_residual_growth` limit is activated. Default: ``100``.

.. note::

   :class:`LineSearch` can only be set when the :py:attr:`~NavierStokesSolver.linear_solver` is a :class:`KrylovLinearSolver`.

Krylov Solver Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following recommendations are provided for tuning the Krylov solver:

- **Start with defaults:** Begin with the default settings (:py:attr:`~KrylovLinearSolver.max_iterations` = 15, :py:attr:`~KrylovLinearSolver.max_preconditioner_iterations` = 25, :py:attr:`~KrylovLinearSolver.relative_tolerance` = 0.05).

- **Improving convergence:** If convergence is slow, first try increasing :py:attr:`~KrylovLinearSolver.max_preconditioner_iterations` (values between 10–35 are typically effective). If that is insufficient, increase :py:attr:`~KrylovLinearSolver.max_iterations`.

- **Improving performance:** If the case converges well, try reducing :py:attr:`~KrylovLinearSolver.max_iterations` to reduce time per pseudo-step.

- **Low Mach number flows:** Cases with low Mach numbers may require a tighter :py:attr:`~KrylovLinearSolver.relative_tolerance`. Try reducing it to values as low as ``1e-3`` if convergence is difficult.

- **Kappa MUSCL:** Setting :py:attr:`~NavierStokesSolver.kappa_MUSCL` to ``0.33`` has been observed to cause residual stalling at final convergence levels when using the Krylov solver. If this occurs, try values of ``0`` or ``-0.33`` instead.

- **Line search:** If the CFL is not increasing, try increasing the :py:attr:`~LineSearch.residual_growth_threshold` (e.g., from ``0.85`` toward ``1.0``) to allow more residual growth per step. If residuals stall, oscillate, or the CFL grows large without making progress, reduce the threshold to enforce stricter convergence per step.

- **Turbulence solver interaction:** Because the Krylov solver significantly reduces the number of Navier-Stokes pseudo-steps needed, the turbulence equations can become the limiting factor for overall convergence. The automatic evaluation frequency adjustment (6 turbulence updates per Navier-Stokes update) addresses this, but if the turbulence solver stalls, it may limit overall convergence. Monitoring both Navier-Stokes and turbulence residuals is recommended.

.. TODO: Need changing once AMGx (incompressible or CHT) is ready

:py:attr:`~NavierStokesSolver.update_jacobian_frequency`
--------------------------------------------------------

The default value for :py:attr:`~NavierStokesSolver.update_jacobian_frequency` is 4, which means that the Jacobian for evaluating the NS equation is updated every 4 pseudo-steps. For some challenging cases, reducing :py:attr:`~NavierStokesSolver.update_jacobian_frequency` from 4 to 1 may help, however, this may slow the NS solver by up to approximately 30%.

.. note::

   When using the :class:`KrylovLinearSolver`, the Navier-Stokes Jacobian update frequency is automatically set to 6, and the turbulence model Jacobian update frequency is set to 1. These values are managed by the solver and do not need to be configured manually.

:py:attr:`~NavierStokesSolver.equation_evaluation_frequency`
------------------------------------------------------------
The default value for :py:attr:`~NavierStokesSolver.equation_evaluation_frequency` is 1, which means that the Navier-Stokes solution is updated every pseudo-step. For loosely-coupled simulations, the :py:attr:`~NavierStokesSolver.equation_evaluation_frequency` value can be changed to introduce a solution update at a different frequency than the turbulence/transition model solvers. The recommended value for this parameter is 1 for a large majority of simulations.

.. note::

   When using the :class:`KrylovLinearSolver`, the Navier-Stokes equation evaluation frequency is automatically set to 6, while the turbulence model equation evaluation frequency is set to 1. This results in 6 turbulence updates per Navier-Stokes update, helping both equation systems converge at a similar rate. These values are managed by the solver and do not need to be configured manually.

.. _knowledge_base_riemannSolver:

:py:attr:`~NavierStokesSolver.riemann_solver`
---------------------------------------------

The :py:attr:`~NavierStokesSolver.riemann_solver` selects the Riemann solver used for the inviscid flux. Two options are available:

- :class:`RoeFlux` (default): the scheme used for the large majority of subsonic and transonic simulations.
- :class:`SLAU2Flux`: an alternative scheme that can improve robustness for supersonic and high-speed flows.

.. note::

   SLAU2 is applied to the interior face fluxes only. Boundary fluxes (freestream, supersonic inflow, wall, and so on) always use the Roe flux, even when :class:`SLAU2Flux` is selected.

Jacobian
~~~~~~~~

When :class:`SLAU2Flux` is selected, the :py:attr:`~SLAU2Flux.jacobian` option selects the Jacobian formulation: ``"SLAU2"`` (default) or ``"Roe"``.

.. _knowledge_base_lowDissipationScheme:

:py:attr:`~NavierStokesSolver.numerical_dissipation_factor`
-----------------------------------------------------------

The low-dissipation Roe scheme in Flow360 is a modification of the Roe scheme designed to address low Mach number problems and achieve reduced numerical dissipation in the range of higher-resolved wave numbers.

For steady simulations with the low-dissipation scheme, it is strongly recommend **NOT** to use Ramp CFL and use adaptive CFL instead due to divergence issue. For flow conditions with low Mach numbers and low Reynolds numbers, it is more effective.

The low-dissipation parameter determines the reduction in numerical flux dissipation. The recommended value for this parameter is 0.2. However, to achieve better convergence, a value of 0.5 can be used.

The solver setup parameters for the low-dissipation feature are described below.

Solver setup recommendation for the low-dissipation scheme
------------------------------------------------------------

The following recommendations are provided to assist in running simulations with the low-dissipation scheme:

- It is recommended to first run a steady or unsteady simulation, and from that solution, start a simulation with the low-dissipation scheme.

- It is recommended to achieve a two-order-of-magnitude reduction in nonlinear residuals and keep the linear residual below 5 when running a simulation with the low-dissipation scheme. To achieve this, the :py:attr:`~Unsteady.step_size` can be halved. Additionally, it is advisable to slightly increase the :py:attr:`Unsteady.max_pseudo_steps` / :py:attr:`Steady.max_steps`, and :py:attr:`~LinearSolver.max_iterations` parameter in the :py:attr:`~NavierStokesSolver.linear_solver`.

- Since the low-dissipation scheme contributes to high-fidelity simulation, it is recommended to use a second-order spatial discretization with the low-dissipation scheme by setting the :py:attr:`~NavierStokesSolver.order_of_accuracy` to 2 for both the Navier-Stokes and turbulence solvers. Alternatively, the first-order option can be used to initially march in time and space and create a well-developed initial solution before switching to the second-order scheme. This is beneficial when the flow field needs to be fully developed throughout the domain in an unsteady simulation.

- To achieve accuracy between 2nd and 3rd order in the solution, it is recommended to use a value of 1/3 for :py:attr:`~NavierStokesSolver.kappa_MUSCL` with the low-dissipation scheme.

- In the case of poor convergence, the :py:attr:`~NavierStokesSolver.numerical_dissipation_factor` can be increased to 0.5 or a value higher than that. Additionally, when facing such issues, setting the :py:attr:`~NavierStokesSolver.kappa_MUSCL` parameter to -1 may also be helpful.

- It is recommended to use a value of 1 for the :py:attr:`~NavierStokesSolver.update_jacobian_frequency` with the low-dissipation scheme.

- For CFL ramping when using the low-dissipation scheme, it is recommended to set a high value (such as 1e+5) for the CFL number and set the :py:attr:`~RampCFL.ramp_steps` to 1. Both the initial and final CFL values can be set to the same number to keep the CFL number fixed. By using a high fixed CFL number and a small :py:attr:`~Unsteady.step_size`, the required values for :py:attr:`~Unsteady.max_pseudo_steps` and :py:attr:`~LinearSolver.max_iterations` can be reduced to meet the convergence criteria.

- In the case of divergence, it is recommended to reduce the :py:attr:`~Unsteady.step_size` by half until the divergence issue is resolved.
