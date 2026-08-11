.. _runControl:

******************
Run Control
******************

Run control allows you to configure stopping criteria that automatically terminate a simulation when monitored output fields reach specified tolerance thresholds. Stopping criteria provide additional convergence control based on user-defined metrics of interest (such as force coefficients, probe values, or surface probe data), working in conjunction with solver tolerances—both must be satisfied for the simulation to stop.

.. important::
   Stopping criteria monitor outputs that must be configured in the :ref:`Output Configuration <outputConfiguration>` section. Ensure your desired outputs (force outputs, probe outputs, or surface probe outputs) are set up before configuring stopping criteria.

How Stopping Criteria Work
===========================

When you configure one or more stopping criteria:

1. **Field Computation**: The specified output field is computed from the selected output type. For Force Outputs, this includes force and moment coefficients (such as lift, drag, and pitching moment). For Probe Outputs, this includes flow variables at point locations (such as pressure, Mach number, or velocity). For Surface Probe Outputs, this includes flow variables projected onto surfaces (such as pressure coefficient or skin friction). All these outputs are processed using the moving statistics object configured in the output, and this processed data is ultimately fed to assess the stopping criterion. See :ref:`Python API Outputs <outputs>` for the complete definition of these settings.

2. **Combined Criteria Evaluation**: Stopping criteria provide additional convergence control based on user-defined metrics. The simulation stops when **both**:

   * All user-defined stopping criteria are satisfied (if a ``tolerance_window_size`` is set, the criterion checks that ``|max − min| / 2`` of the monitored field within that window is below the ``tolerance``; if no window is set, the latest value is compared directly against the tolerance), **and**
   * The solver residual tolerances are met (Navier-Stokes solver, turbulence solver, and transition model solver residuals are within their specified tolerances).

   .. note:: **What counts as a "data point"**

      ``tolerance_window_size`` counts **data points** from the monitor output, not solver
      iterations:

      * **Steady simulations.** The solver writes one data point every 10 pseudo-steps, so a
        ``tolerance_window_size`` of :math:`N` checks the last :math:`10 N` pseudo-steps.
      * **Time-accurate (unsteady) simulations.** Only the final value of each physical step
        contributes to the window. A ``tolerance_window_size`` of :math:`N` checks the last
        :math:`N` physical (time) steps.

      The same convention applies to ``MovingStatistic.moving_window_size`` — see
      :class:`~flow360.MovingStatistic` for details.

3. **Simulation Stop**: The simulation stops when all configured criteria are satisfied simultaneously. However, the maximum number of steps (set in time stepping configuration) has the highest authority and will stop the simulation regardless of other conditions.

.. note:: **Residual convergence — steady vs. unsteady**

   The residual convergence requirement operates differently depending on the simulation type:

   * **Steady state**: only pseudo-steps are executed (effectively a single infinite physical step). The simulation terminates when the residuals drop below the **absolute tolerance** and all user-defined stopping criteria are satisfied.
   * **Unsteady**: at each physical step, up to ``max_pseudo_steps`` pseudo-steps are executed. The solver advances to the next physical step either when ``max_pseudo_steps`` is reached or when the residuals satisfy the solver tolerance — whichever comes first. Solver tolerances for unsteady simulations can be specified as:

     - **Absolute tolerance**: the residual falls below a fixed threshold.
     - **Relative tolerance**: the residual falls below a fraction of the initial residual of that physical step (e.g., ``relative_tolerance = 1e-2`` means two orders of magnitude reduction).

.. important::
   There is an AND relation between user-defined stopping criteria and solver residual tolerances—both must be satisfied for the simulation to stop. Satisfying the stopping criteria alone does not halt the simulation; the solver residuals must also meet their convergence requirements. Conversely, if the solver tolerances are met first, the simulation continues until all user-defined stopping criteria are satisfied. The maximum number of steps (set in time stepping configuration) overrides all other stopping conditions and will terminate the simulation when reached.

See Also
========

* :ref:`Python API Run Control <python_api_run_control>` - Complete API documentation for ``StoppingCriterion`` and ``RunControl`` classes
* :ref:`GUI Guide: Stopping Criteria <stopping_criteria>` - GUI-specific documentation
