.. _knowledge_base_turbulenceModelSolver:

.. currentmodule:: flow360

TurbulenceModelSolver
=====================

Turbulence Model Types
----------------------

:class:`SpalartAllmaras` activates the Spalart-Allmaras (SA) turbulence model which is the most widely used in external aerodynamics applications. This is highlighted by the use of different turbulence models in the AIAA High Lift and Drag Prediction Workshops, where the SA model dominates.  The SA turbulence model involves solving a single transport equation for a value known as the Spalart variable (similar to eddy viscosity). As the model is a single-equation turbulence model, the computational cost for solving this transport equation is low. As the Spalart variable is linear at the wall the turbulence model is also less sensitive to grid resolution near the wall ($y^+$ higher than one is acceptable) compared to many two-equation turbulence models and also leads to higher robustness.

:class:`KOmegaSST` activates the :math:`k-\omega` SST turbulence model of Menter, which was initially developed to improve the predictions under adverse pressure gradients compared to other two-equation turbulence models. The model blends the :math:`k-\omega` model near the wall to the :math:`k-\epsilon` model away from it using a blending function and implements a limiter on the eddy viscosity (based on Bradshaw's assumption) improving turbulent shear stress predictions under adverse pressure gradients. This model is widely used in both external and internal aerodynamics and has shown good correlation with experiments over a wide range of use cases. The :math:`k-\omega` SST model is especially recommended for internal flow applications. The primary advantage of two-equation models over one-equation models (such as the SA model) is the ability of the model to distinguish between small and large scale turbulence. However, as two equations are solved simultaneously with variables of different orders of magnitude, the model is tougher to converge and more expensive computationally than the SA model.

:py:attr:`~SpalartAllmaras.absolute_tolerance`
---------------------------------------------------

The :py:attr:`~SpalartAllmaras.absolute_tolerance` is the primary convergence metric for steady cases. At least 5 orders of magnitude reduction is recommended for all residual values. The :py:attr:`~SpalartAllmaras.absolute_tolerance` can also be used for unsteady cases, but is less meaningful than the :py:attr:`~SpalartAllmaras.relative_tolerance`, as the initial residual values change between different physical steps.

.. _relativeTolerance2:

:py:attr:`~SpalartAllmaras.relative_tolerance`
---------------------------------------------------

The relative residual is defined as the ratio of the current pseudoStep's residual to the maximum residual present in the first 10 pseudoSteps within the current physicalStep. When running unsteady cases, the :py:attr:`~SpalartAllmaras.relative_tolerance` is typically set to 1e-2 or 1e-3. Once the nonlinear residuals drop by 2 or 3 orders of magnitude, the solver will continue to the next physicalStep. The :py:attr:`~SpalartAllmaras.relative_tolerance` is ignored for steady cases.



:py:attr:`~SpalartAllmaras.order_of_accuracy`
--------------------------------------------------

As recommended in the :ref:`orderOfAccuracy of NavierStokesSolver <knowledge_base_orderOfAccuracy>`, when solving unsteady cases, it may be necessary to initialize the flow field with :py:attr:`~SpalartAllmaras.order_of_accuracy` set to 1. Once the flow field has been initialized, the user can create a child case and switch the :py:attr:`~SpalartAllmaras.order_of_accuracy` back to 2. 

When adjusting the :py:attr:`~SpalartAllmaras.order_of_accuracy` for the :code:`TurbulenceModelSolver` (:class:`SpalartAllmaras` or :class:`KOmegaSST`), the :ref:`NavierStokesSolver <knowledge_base_navierStokesSolver>` should be adjusted as well.

:py:attr:`~SpalartAllmaras.linear_solver`
----------------------------------------------------

The turbulence solver is typically easier to converge than the NS solver. Therefore, the value of :py:attr:`~LinearSolver.max_iterations` for the turbulence solver, typically set to ~20, is less than :py:attr:`~LinearSolver.max_iterations` for the NS solver. However, if the linear residual reduction ratio after linear solver is not enough, increasing :py:attr:`~LinearSolver.max_iterations` up to ~50 could be helpful. The default :py:attr:`~LinearSolver.max_iterations` for turbulence solver is 20.

.. 
    [TODO] For challenging cases see the :ref:debug divergence subsection <XXXXX>

:py:attr:`~SpalartAllmaras.update_jacobian_frequency`
----------------------------------------------------------

Similar to the NS solver, the default value for :py:attr:`~SpalartAllmaras.update_jacobian_frequency` is 4, indicating that the Jacobian for evaluating the turbulence equation is only updated every 4 pseudo-steps. For more challenging cases, :py:attr:`~SpalartAllmaras.update_jacobian_frequency` may need to be reduced from 4 to 1. This will not significantly slow down the solver, since the turbulence equation is not as computationally expensive as the NS equation.

.. 
    [TODO] For challenging cases see the :ref:debug divergence subsection <XXXXX>

:py:attr:`~SpalartAllmaras.equation_evaluation_frequency`
---------------------------------------------------------------------

As mentioned above, the turbulence equation is typically easier to converge than the NS equations.
Therefore, by default, :py:attr:`~SpalartAllmaras.equation_evaluation_frequency` is set to 4, meaning that the turbulence equation is only evaluated every 4 pseudo-steps. For challenging cases, :py:attr:`~SpalartAllmaras.equation_evaluation_frequency` may need to be reduced from 4 to 1 as well. This change will not significantly impact the solver's performance.


:py:attr:`~SpalartAllmaras.rotation_correction`
----------------------------------------------------

The :py:attr:`~SpalartAllmaras.rotation_correction` activates the rotation-curvature correction and is applicable to both the Spalart-Allmaras and :math:`k-\omega` SST turbulence models. This correction modifies the production term to account for shear and rotation effects on the turbulence intensity. It is recommended to set :py:attr:`~SpalartAllmaras.rotation_correction` to true for flows with significant rotation effects, such as those encountered in turbomachinery and rotorcraft.

:py:attr:`~SpalartAllmaras.quadratic_constitutive_relation`
---------------------------------------------------------------

The :py:attr:`~SpalartAllmaras.quadratic_constitutive_relation` activates the quadratic constitutive relation for the turbulence shear stress tensor which accounts for anisotropy. This correction leads to improved predictions for corner flow separation and juncture flows. The correction is applicable to both the Spalart-Allmaras and :math:`k-\omega` SST turbulence models. 

:py:attr:`~SpalartAllmaras.hybrid_model`
------------------------------------------

The :py:attr:`~SpalartAllmaras.hybrid_model` option activates hybrid RANS-LES turbulence models and is only valid for unsteady flows. This option is recommended for cases with more complex flow physics (significant separation regions, bluff body flows), leading to higher solution fidelity compared to pure RANS solutions. Hybrid models blend the use of RANS-based models near the wall with LES-based modelling away from the wall, and therefore, lead to significantly reduced grid and time step requirements compared to simulations that are purely LES based. This means that large scale turbulence away from the wall is no longer modelled but directly resolved based on the distance from the wall and grid resolution, often leading to a reduced turbulent length scale and turbulent viscosity. :py:attr:`~SpalartAllmaras.hybrid_model` can be used with both the Spalart-Allmaras and :math:`k-\omega` SST turbulence models.

Two shielding functions are available via :py:attr:`~DetachedEddySimulation.shielding_function`:

- **DDES (Delayed Detached Eddy Simulation)** (Spalart 2006): Modifies the turbulence length scale to delay the switch from RANS to LES, preventing premature LES activation in attached boundary layers. Shielding can break down when the lateral grid spacing in the boundary layer :math:`d_x` satisfies :math:`d_x / \delta < 0.3` (where :math:`\delta` is the local boundary-layer thickness), which may cause grid-induced separation.
- **ZDES (Zonal Detached Eddy Simulation)** (Deck and Renard 2020): An enhanced hybrid RANS/LES approach that employs dual shielding functions and an inhibition mechanism based on wall-normal gradients. This prevents premature transition from RANS to LES regardless of mesh refinement or adverse pressure gradients while ensuring a rapid switch to LES in free shear layers. ZDES is more tolerant to fine lateral grid spacing than DDES.

The LES filter width in hybrid RANS-LES regions is controlled by :py:attr:`~DetachedEddySimulation.grid_size_for_LES`, which offers three options:

- **maxEdgeLength** (default): Uses the longest edge of each cell as the LES length scale. Suitable for anisotropic meshes and general-purpose hybrid RANS-LES simulations.
- **meanEdgeLength**: Uses the arithmetic mean of all cell edge lengths. Suitable for isotropic grids, providing a slightly reduced filter width compared to ``maxEdgeLength``.
- **shearLayerAdapted**: Uses a shear-layer-adapted length scale that accounts for the local flow direction within the shear layer rather than relying solely on cell geometry. This option is recommended for flows with prominent free shear layers — such as mixing layers, jet flows, and massively separated wakes — where standard grid-based length scales can over-estimate the filter width and suppress the resolved turbulent energy that the hybrid model is designed to capture.

.. 
    [TODO] For challenging cases see the :ref:debug divergence subsection <XXXXX>
