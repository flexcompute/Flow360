.. _knowledge_base_heatEquationSolver:

.. currentmodule:: flow360

HeatEquationSolver
==================

:py:attr:`~HeatEquationSolver.equation_evaluation_frequency`
----------------------------------------------------------------------------------
For steady simulations the default value for :py:attr:`~HeatEquationSolver.equation_evaluation_frequency` is 10, which means if the flow solver is evaluated at every pseudo step, the heat solver is evaluated every 10 pseudo steps of the flow solution.
This parameter essentially controls the feedback frequency from the solid to the fluid. 
It should always be greater than or equal to the :py:attr:`~HeatEquationSolver.equation_evaluation_frequency` of the flow solver.

:py:attr:`~HeatEquationSolver.linear_solver`
----------------------------------------------------

:py:attr:`~HeatEquationSolver.linear_solver` controls the settings of the linear solver. Either :py:attr:`~HeatEquationSolver.absolute_tolerance` or :py:attr:`~HeatEquationSolver.relative_tolerance` can be used to determine the convergence of the linear solver.
Together with :py:attr:`~LinearSolver.max_iterations`, they determine how much the heat equation solver residual converges in each pseudo step.
