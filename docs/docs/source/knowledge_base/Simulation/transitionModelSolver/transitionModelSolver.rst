.. _knowledge_base_transitionModelSolver:

.. currentmodule:: flow360

TransitionModelSolver
=====================

The laminar to turbulence transition model supported by Flow360 is the 2019b version of the Amplification Factor Transport (AFT) model created by James Coder, University of Tennessee.
This model adds two additional equations to the flow solver in order to solve for the amplification factor and intermittency flow quantities.
More details about the model can be found `here <https://tmbwg.github.io/turbmodels/aft_transition_3eqn.html>`_.
Below are a list of configuration parameters for the transition model.
Either :py:attr:`~TransitionModelSolver.N_crit` or :py:attr:`~TransitionModelSolver.turbulence_intensity_percent` can be used to tune the location of transition from laminar to turbulent flow.

:py:attr:`~TransitionModelSolver.absolute_tolerance`
---------------------------------------------------------------------------

The :py:attr:`~TransitionModelSolver.absolute_tolerance` is the primary convergence metric for steady cases. At least 5 orders of magnitude reduction is recommended for all residual values. The :py:attr:`~TransitionModelSolver.absolute_tolerance` can also be used for unsteady cases, but is less meaningful than the :py:attr:`~TransitionModelSolver.relative_tolerance`, as the initial residual values change between different physical steps.

.. _relativeTolerance3:

:py:attr:`~TransitionModelSolver.relative_tolerance`
---------------------------------------------------------------------------

The relative residual is defined as the ratio of the current pseudoStep's residual to the maximum residual present in the first 10 pseudoSteps within the current physicalStep. When running unsteady cases, the :py:attr:`~TransitionModelSolver.relative_tolerance` is typically set to 1e-2 or 1e-3. Once the nonlinear residuals drop by 2 or 3 orders of magnitude, the solver will continue to the next physicalStep. The :py:attr:`~TransitionModelSolver.relative_tolerance` is ignored for steady cases.



:py:attr:`~TransitionModelSolver.order_of_accuracy`
--------------------------------------------------------------------------

As recommended in the :ref:`orderOfAccuracy of navierStokesSolver <knowledge_base_orderOfAccuracy>`, when solving unsteady cases, it may be necessary to initialize the flow field with :py:attr:`~TransitionModelSolver.order_of_accuracy` set to 1. Once the flow field has been initialized, the user can create a child case and switch the :py:attr:`~TransitionModelSolver.order_of_accuracy` back to 2. 

When adjusting the :py:attr:`~TransitionModelSolver.order_of_accuracy` for the :class:`TransitionModelSolver`, the :ref:`navierStokesSolver <knowledge_base_navierStokesSolver>` and :ref:`turbulenceModelSolver <knowledge_base_turbulenceModelSolver>` should be adjusted as well.

:py:attr:`~TransitionModelSolver.linear_solver`
---------------------------------------------------

The transition solver is typically easier to converge than the NS solver. Therefore, the value of :py:attr:`~LinearSolver.max_iterations` for the transition solver, typically set to ~20, is less than :py:attr:`~LinearSolver.max_iterations` for the NS solver. However, if the linear residual reduction ratio after linear solver is not enough, increasing :py:attr:`~LinearSolver.max_iterations` up to ~50 could be helpful. The default :py:attr:`~LinearSolver.max_iterations` for transition solver is 20.

.. 
    [TODO] For challenging cases see the :ref:debug divergence subsection <XXXXX>

:py:attr:`~TransitionModelSolver.update_jacobian_frequency`
----------------------------------------------------------------------------------

Similar to the NS solver, the default value for :py:attr:`~TransitionModelSolver.update_jacobian_frequency` is 4, indicating that the Jacobian for evaluating the transition equations is only updated every 4 pseudo-steps. For more challenging cases, :py:attr:`~TransitionModelSolver.update_jacobian_frequency` may need to be reduced from 4 to 1. This will not significantly slow down the solver, since the transition equation is not as computationally expensive as the NS equation.

.. 
    [TODO] For challenging cases see the :ref:debug divergence subsection <XXXXX>

:py:attr:`~TransitionModelSolver.equation_evaluation_frequency`
--------------------------------------------------------------------------------

As mentioned above, the transition equation is typically easier to converge than the NS equations.
Therefore, by default, :py:attr:`~TransitionModelSolver.equation_evaluation_frequency` is set to 4, meaning that the transition equation is only evaluated every 4 pseudo-steps. For challenging cases, :py:attr:`~TransitionModelSolver.equation_evaluation_frequency` may need to be reduced from 4 to 1 as well. This change will not significantly impact the solver's performance.

.. 
    [TODO] For challenging cases see the :ref:debug divergence subsection <XXXXX>

.. _Ncrit:


:py:attr:`~TransitionModelSolver.N_crit`
---------------------------------------------------------------

:py:attr:`~TransitionModelSolver.N_crit` is the critical amplification factor. Boundary layer transition is triggered when the amplified frequency of the Tollmien-Schlichting waves reaches this value. Higher values delay the onset of laminar-turbulent transition. Only :py:attr:`~TransitionModelSolver.N_crit` or :py:attr:`~TransitionModelSolver.turbulence_intensity_percent`, can be specified in the flow configuration file. The value has a range of 1 to 11 in Flow360. 

.. _TurbI:

:py:attr:`~TransitionModelSolver.turbulence_intensity_percent`
------------------------------------------------------------------------------------

:py:attr:`~TransitionModelSolver.turbulence_intensity_percent` is used to compute the :py:attr:`~TransitionModelSolver.N_crit` parameter (see above) when :py:attr:`~TransitionModelSolver.N_crit` is not set for the AFT transition model using: :math:`N_{crit} = -8.43 - 2.4*ln (0.025*tanh(\text{turbulenceIntensityPercent}/2.5))`. Higher values of :py:attr:`~TransitionModelSolver.turbulence_intensity_percent` therefore lead to earlier transition. The value of :py:attr:`~TransitionModelSolver.turbulence_intensity_percent` has a range of 0.03 to 2.5 (%) in Flow360.
