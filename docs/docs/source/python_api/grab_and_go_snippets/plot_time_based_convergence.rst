.. _python_api_plot_probe_time_history:

.. currentmodule:: flow360

*******************************
Plot Probe Monitor Time History
*******************************

This example demonstrates how to plot a time history of pressure from a probe monitor.

.. literalinclude:: _snippets/plot_time_based_convergence.py
   :language: python

Notes
=====

- Use ``case.results.monitors.monitor_names`` to list all available monitors.
- Access monitors by name using ``case.results.monitors["name"]``.
- Call ``reload_data(include_time=True)`` to add a time column for unsteady simulations.
- The ``as_dataframe()`` method returns a pandas DataFrame for easy plotting and analysis.
- Use the names from the output configuration to construct the column name for the plot.
