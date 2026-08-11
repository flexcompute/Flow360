.. _python_API_monitor_conversion:

.. currentmodule:: flow360

**************************
Example Monitor Conversion
**************************

This example demonstrates the conversion of legacy monitor point definitions, previously specified in a JSON format (v0), into the corresponding `ProbeOutput` objects compatible with the `SimulationParams` class used in the current Flow360 Python API. The script employs the helper function ``ProbeOutput.read_all_v0_monitors`` for this conversion process.

.. literalinclude:: ../../../../../examples/migration_guide/example_monitor_conversion.py
  :language: python
  :linenos:

Notes
=====

- The script utilizes ``ProbeOutput.read_all_v0_monitors`` from the ``flow360.component.simulation.migration`` module to facilitate the transition of monitor point definitions from the legacy JSON format.
- The resulting list of converted monitor objects can be directly incorporated into the ``outputs`` list within a ``SimulationParams`` instance.
