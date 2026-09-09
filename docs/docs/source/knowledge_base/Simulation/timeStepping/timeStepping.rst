.. _knowledge_base_timeStepping:

.. currentmodule:: flow360

Time Stepping
=============

:code:`timeStepSize`
---------------------------------------

:code:`timeStepSize` prescribes the size of each physical time step.

- For steady case, the steady solver can be activated using :class:`Steady` class without setting :code:`timeStepSize`.

- For unsteady cases, there are two typical ways for determining the :py:attr:`~flow360.Unsteady.step_size`:

  - Based on an expected :ref:`vortex shedding frequency <vortex_shedding_freq>` 
  - Based on the desired :ref:`angle of rotation per step <rot_angle_per_step>`

If a single volume zone with a non-zero :code:`omega` is present in the simulation, setting :py:attr:`SimulationParams.time_stepping` with :class:`Steady` activates a Single Reference Frame simulation. For multiple volume zones with different :code:`omega` values, the Multiple Reference Frame will be used.

.. _vortex_shedding_freq:

Vortex Shedding Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^

The vortex shedding frequency can be estimated by:

.. math::

    f = \frac{St \cdot U_\infty}{D }


where :math:`U_\infty` is the freestream speed,
:math:`D` is the characteristic length, for example the mean aerodynamic chord,
and :math:`St` is the `Strouhal Number <https://en.wikipedia.org/wiki/Strouhal_number>`_. An approximation of :math:`St = 0.2` is typically appropriate for most CFD applications.

Typically, the physical time step size :math:`\Delta t` is set as:

.. math::

    \Delta t = \frac{1}{100} \cdot \frac{1}{f}

.. _rot_angle_per_step:

Angle of Rotation per Step 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If there is a transient BET Line or a sliding interface modeled, then the :code:`timeStepSize` can be calculated from the desired angle of rotation in each physical step.

.. table:: Recommended angle of rotation per step for different scenarios.
   :widths: 50 40

   +------------------------------------------------------+---------------------------------+
   | Scenario                                             | Angle per :code:`timeStepSize`  |
   +======================================================+=================================+
   | Transient BETDisks (i.e., BET Line)                  | 6 deg/step                      |
   +------------------------------------------------------+---------------------------------+
   | | Rotor in sliding interface, phase-I:               | 6 deg/step                      |
   | | Initializing the flow-field with 1st-order solver  |                                 |
   +------------------------------------------------------+---------------------------------+
   | | Rotor in sliding interface, phase-II:              | 6 deg/step                      |
   | | Initializing the flow-field with 2nd-order solver  |                                 |
   +------------------------------------------------------+---------------------------------+
   | | Rotor in sliding interface, phase-III:             | 3 deg/step                      |
   | | Collecting the data with 2nd-order solver          |                                 |
   +------------------------------------------------------+---------------------------------+
   | Strong blade-vortex interaction (hover/descent)      | 1-2 deg/step                    |
   +------------------------------------------------------+---------------------------------+
   | Generating high-quality time-resolved animation      | 0.5-1 deg/step                  |
   +------------------------------------------------------+---------------------------------+

For more information about 1st versus 2nd order solution, see :ref:`orderOfAccuracy <knowledge_base_orderOfAccuracy>`. 
Once the angle per step is determined, then the :code:`timeStepSize` can be calculated:

.. math::

   \text{timeStepSize} = \theta \cdot \frac{\pi}{180} \cdot \frac{1}{\text{omegaRadians}}

where :math:`\theta` is the angle of rotation in degrees. Note that the :code:`omegaRadians` here is the nondimensional rotation speed,
which can be :ref:`easily converted from RPM <non_dim_omega>`.

.. _knowledge_base_maxPseudoSteps:

:code:`maxPseudoSteps`
-----------------------

- For steady cases, :py:attr:`Steady.max_steps` is typically 5K~10K. If the nonlinear residuals and/or the forces and moments are still changing, the user can always fork the completed case and let it run more pseudo steps.
- For unsteady time-accurate cases, :py:attr:`~Unsteady.max_pseudo_steps` is typically slightly larger than :py:attr:`~RampCFL.ramp_steps` so the :py:attr:`~RampCFL.final` CFL can be achieved. As shown in the examples below, when :py:attr:`~RampCFL.ramp_steps` = 10 then :py:attr:`~Unsteady.max_pseudo_steps` should be set to 12. If :py:attr:`~RampCFL.ramp_steps` = 33 then :py:attr:`~Unsteady.max_pseudo_steps` should be set to 35.


Example :py:attr:`~SimulationParams.time_stepping` for the 1st-order unsteady cases
  .. code-block:: python

   time_stepping = fl.Unsteady(
      max_pseudo_steps=12, CFL=fl.RampCFL(initial=1, final=1000, ramp_steps=10)
   )

Example :py:attr:`~SimulationParams.time_stepping` for the 2nd-order unsteady cases
  .. code-block:: python

   time_stepping = fl.Unsteady(
      max_pseudo_steps=35, CFL=fl.RampCFL(initial=1, final=1e7, ramp_steps=33)
   )

For more information about 1st versus 2nd order solution, see :ref:`orderOfAccuracy <knowledge_base_orderOfAccuracy>`.
The CFL number will be discussed in the following subsection.

.. _knowledge_base_CFL:

:code:`CFL`
------------

The CFL number determines the pseudo step size in each physical step. Two different approaches exist in Flow360 to drive the CFL number (which are activated by setting the :py:attr:`Steady.CFL` or :py:attr:`Unsteady.CFL`):
 
- :class:`RampCFL`, where the CFL ramping is defined by the user.
- :class:`AdaptiveCFL`, where the ramping is automatically adapted based on the linear residual values. 
  
:class:`RampCFL`
^^^^^^^^^^^^^^^^
The CFL ramping is defined by the user using this approach by setting values for :py:attr:`~RampCFL.initial` , :py:attr:`~RampCFL.final` and :py:attr:`~RampCFL.ramp_steps`. The table below shows recommended values for CFL number ramping for steady and unsteady cases.

.. list-table:: Recommended CFL ramp up for steady and unsteady cases.
   :widths: 42 5 22 22 5
   :header-rows: 1

   * - Example
     - Type
     - :py:attr:`~RampCFL.initial`
     - :py:attr:`~RampCFL.final`
     - :py:attr:`~RampCFL.ramp_steps`
   * - Simple wing or fuselage, mostly attached linear flow
     - Steady
     - 5 
     - 200
     - 100
   * - Full aircraft with nonlinear flow, some separation, simple actuator/BET disks

     - Steady
     - 1
     - 100~150
     - 1K~2K
   * - Full aircraft near onset of stall, large-scale separation, challenging actuator/BET disks

     - Steady
     - 0.1~1
     - 10~50
     - 3K~5K
   * - Forked/child case
     - Steady
     - parent :py:attr:`~RampCFL.final` CFL
     - parent :py:attr:`~RampCFL.final` CFL
     - 1
   * - Rotor in sliding interface, 1st-order solver
     - Unsteady
     - 1
     - 1000
     - ~10
   * - Rotor in sliding interface, 2nd-order solver
     - Unsteady
     - 1
     - 1e+5~1e+7
     - 30~50

#. For steady cases, :py:attr:`~RampCFL.initial` CFL < 1, :py:attr:`~RampCFL.final` CFL < 100, :py:attr:`~RampCFL.ramp_steps` > 3K are considered as conservative. Such conservative values are only recommended for challenging cases.
#. For unsteady cases, it is typically recommended to let the nonlinear residuals drop 2~3 orders of magnitude in each physical step. That is why higher :py:attr:`~RampCFL.final` CFL and more aggressive CFL ramping values are suggested.

#. For unsteady cases running 2nd-order solver, :py:attr:`~RampCFL.final` CFL < 1e+5, :py:attr:`~RampCFL.ramp_steps` > 50 are considered as conservative. Decreasing :py:attr:`~RampCFL.ramp_steps` and :py:attr:`~Unsteady.max_pseudo_steps` within each physical step will decrease the overall cost and runtime. For more information about 1st versus 2nd order solutions, see :ref:`orderOfAccuracy <knowledge_base_orderOfAccuracy>`.

.. _knowledge_base_adaptive_CFL:


:class:`AdaptiveCFL`
^^^^^^^^^^^^^^^^^^^^

This approach adaptively adjusts the CFL number with criteria based on the linear residual values. The :class:`AdaptiveCFL` ramping approach offers several advantages when compared to :class:`RampCFL`:

- The user does not need prior knowledge on the complexity of the simulation in terms of how conservatively or aggressively the CFL can be driven.
- The :class:`AdaptiveCFL` will lead to faster convergence, when the user enters too conservative CFL settings in :class:`RampCFL`
- The :class:`AdaptiveCFL` can prevent divergence when the user enters too aggressive CFL settings in :class:`RampCFL` (when the divergence is related to the time stepping settings, rather than mesh related). 
- The :class:`AdaptiveCFL` will lead to deeper residual convergence when the user enters too aggressive CFL settings in :class:`RampCFL`, but the simulation does not diverge.

A few aspects must be noted when using :class:`AdaptiveCFL`:

- The :class:`AdaptiveCFL` compromises solution speed and solution robustness over a wide range of cases in terms of complexity. Therefore, the :class:`AdaptiveCFL` will not necessarily lead to a faster solution than :class:`RampCFL`, when the :class:`RampCFL` settings are optimised.
- If the linear residuals are low (below 0.1) when divergence occurs during :class:`RampCFL`, switching to :class:`AdaptiveCFL` will not necessarily remedy the issue, as the CFL is adapted based on the linear residual values

As the :class:`AdaptiveCFL` numerical parameters have been optimised over a range of cases in terms of complexity (from 2D aerofoil to complex 3D flows with separation and shocks), 
a number of optional inputs have been made available to the user to alter the :class:`AdaptiveCFL` settings. These include :py:attr:`~AdaptiveCFL.max_relative_change`, 
:py:attr:`~AdaptiveCFL.convergence_limiting_factor`, :py:attr:`~AdaptiveCFL.min` and :py:attr:`~AdaptiveCFL.max`. 

:py:attr:`~AdaptiveCFL.max_relative_change`
..............................................

The :py:attr:`~AdaptiveCFL.max_relative_change` parameter controls the maximum relative change of the CFL number at each pseudo-step, with the default value set to 1.0. 
Increasing this parameter will therefore let the CFL change more rapidly, speeding up convergence but potentially leading to divergence for more complex cases. 
Conversely, reducing the value of this parameter constrains the relative change of the CFL number. 
An example of how the :py:attr:`~AdaptiveCFL.max_relative_change` parameter affects the convergence is shown, with the CFL number, 
non-linear and linear residuals (continuity) convergence shown in :ref:`Fig1_MaxRel_CFLNS`-:ref:`Fig4_MaxRel_NonLinConv`. 
The :py:attr:`~AdaptiveCFL.max_relative_change` value will affect the convergence speed without affecting the thresholds put on the linear residual in the adaptive CFL algorithm at the initial stages of the run. 
The same final CFL values will be reached for different :py:attr:`~AdaptiveCFL.max_relative_change` entries at the end of the simulation (if the simulations are ran long enough). 
If the convergence is slower than expected :class:`RampCFL` settings, it is recommended to increase the :py:attr:`~AdaptiveCFL.max_relative_change` value.

.. _Fig1_MaxRel_CFLNS:
.. figure:: Figures/CFL_NS_MaxRelativeChange.png
   :align: center
   :width: 80%

   Navier-Stokes CFL number during the simulation for a range of :py:attr:`~AdaptiveCFL.max_relative_change` values.
.. _Fig2_MaxRel_CFLSA:
.. figure:: Figures/CFL_SA_MaxRelativeChange.png
   :align: center
   :width: 80%

   Spalart-Allmaras CFL number during the simulation for a range of :py:attr:`~AdaptiveCFL.max_relative_change` values.
.. _Fig3_MaxRel_LinConv:
.. figure:: Figures/Cont_Linear_MaxRelativeChange.png
   :align: center
   :width: 80%

   Continuity linear residual convergence during the simulation for a range of :py:attr:`~AdaptiveCFL.max_relative_change` values.
.. _Fig4_MaxRel_NonLinConv:
.. figure:: Figures/Cont_Nonlinear_MaxRelativeChange.png
   :align: center
   :width: 80%

   Continuity non-linear residual convergence during the simulation for a range of :py:attr:`~AdaptiveCFL.max_relative_change` values.


:py:attr:`~AdaptiveCFL.convergence_limiting_factor`
.....................................................

The :py:attr:`~AdaptiveCFL.convergence_limiting_factor` modifies how conservative or aggressive the limitations on the values of CFL are. 
Smaller  :py:attr:`~AdaptiveCFL.convergence_limiting_factor` values lead to a more conservative CFL value, whereas larger  :py:attr:`~AdaptiveCFL.convergence_limiting_factor` values lead to a more aggressive CFL value. 
The default value is set to 0.25 which is a good compromise between solution speed and robustness across a wide range of cases. 
An example of how the :py:attr:`~AdaptiveCFL.convergence_limiting_factor` parameter affects the convergence is shown, 
with the CFL number, non-linear and linear residuals (continuity) convergence shown in :ref:`Fig5_ConvFactor_CFLNS`-:ref:`Fig8_ConvFactor_NonLinConv`. The :py:attr:`~AdaptiveCFL.convergence_limiting_factor` 
does not affect the initial CFL values (under 250 pseudo-steps in this case), but has an impact on the thresholds of the linear residuals during the adaptive CFL algorithm and CFL numbers reached in the later stages of the simulation. 
Higher values of :py:attr:`~AdaptiveCFL.convergence_limiting_factor` lead to faster convergence of the nonlinear residual but also permit higher values of the linear residual during the simulation. 
In the case of simulation divergence, it is recommended to reduce the value of :py:attr:`~AdaptiveCFL.convergence_limiting_factor` (after checking the mesh quality at the divergence location, see :ref:`debug_divergence_general_recommendations`.

.. _Fig5_ConvFactor_CFLNS:
.. figure:: Figures/CFL_NS_ConvFactor.png
   :align: center
   :width: 80%

   Navier-Stokes CFL number during the simulation for a range of :py:attr:`~AdaptiveCFL.convergence_limiting_factor` values.
.. _Fig6_ConvFactor_CFLSA:
.. figure:: Figures/CFL_SA_ConvFactor.png
   :align: center
   :width: 80%

   Spalart-Allmaras CFL number during the simulation for a range of :py:attr:`~AdaptiveCFL.convergence_limiting_factor` values.
.. _Fig7_ConvFactor_LinConv:
.. figure:: Figures/Cont_Linear_ConvFactor.png
   :align: center
   :width: 80%

   Continuity linear residual convergence during the simulation for a range of :py:attr:`~AdaptiveCFL.convergence_limiting_factor` values.
.. _Fig8_ConvFactor_NonLinConv:
.. figure:: Figures/Cont_Nonlinear_ConvFactor.png
   :align: center
   :width: 80%

   Continuity non-linear residual convergence during the simulation for a range of :py:attr:`~AdaptiveCFL.convergence_limiting_factor` values.



:py:attr:`~AdaptiveCFL.min`
............................

The :py:attr:`~AdaptiveCFL.min` parameter affects the minimum values of the CFL number for the solved equations during the simulation. 

:py:attr:`~AdaptiveCFL.max`
............................

The :py:attr:`~AdaptiveCFL.max` parameter affects the maximum values of the CFL number for the solved equations during the simulation. An example use case for when this parameter would need to be modified is when the Spalart-Allmaras solver diverges with a non-physical turbulent eddy viscosity. As the Spalart-Allmaras is a one-equation model compared to the five equations that are solved for the Navier-Stokes solver, the linear residuals for the Spalart-Allmaras solution will usually be much lower, and therefore the CFL will be driven to much higher values, which may prompt reducing the :py:attr:`~AdaptiveCFL.max` allowable CFL during the adaptive CFL algorithm. 


Comparison between :class:`RampCFL` and :class:`AdaptiveCFL`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A comparison of the :class:`RampCFL` and :class:`AdaptiveCFL` approaches was performed for a range of settings, with the CFL number, non-linear and linear residuals (continuity) convergence shown in :ref:`Fig9_Ramp_CFLNS`-:ref:`Fig12_Ramp_NonLinConv`.

.. _Fig9_Ramp_CFLNS:
.. figure:: Figures/CFL_NS_RampvsAdapt.png
   :align: center
   :width: 80%

   Comparison of the Navier-Stokes CFL number during the simulation for adaptive and ramped CFL.
.. _Fig10_Ramp_CFLSA:
.. figure:: Figures/CFL_SA_RampvsAdapt.png
   :align: center
   :width: 80%

   Comparison of the Spalart-Allmaras CFL number during the simulation for adaptive and ramped CFL.

.. _Fig11_Ramp_LinConv:
.. figure:: Figures/Cont_Linear_RampvsAdapt.png
   :align: center
   :width: 80%

   Comparison of the continuity linear residual convergence during the simulation for adaptive and ramped CFL.

.. _Fig12_Ramp_NonLinConv:
.. figure:: Figures/Cont_Nonlinear_RampvsAdapt.png
   :align: center
   :width: 80%

   Comparison of the continuity nonlinear residual convergence during the simulation for adaptive and ramped CFL.

Firstly, it must be noticed that divergence is not a key limiter in the CFL values used in this example. For an example where adaptive CFL is used to prevent divergence, see :ref:`debug_divergence_general_recommendations`. 
It can be seen that the CFL values at the end of the simulation and convergence speed are driven by the :py:attr:`RampCFL.final` and :py:attr:`AdaptiveCFL.convergence_limiting_factor` for both algorithms. 
The adaptive CFL algorithm allows for much higher Spalart-Allmaras CFL values. In general similar behaviour is seen between the :class:`RampCFL` and :class:`AdaptiveCFL`, 
with less conservative settings leading to faster convergence and higher linear residual values. 
The key benefit of the :class:`AdaptiveCFL` algorithm is that the CFL will automatically be reduced when the linear residuals are above a given threshold, in many cases preventing divergence.

..
  Once we have the debug divergence subsection done, add a link to that section
  I think some hand drawn schematic illustrating the residual patterns could be very helpful
