.. _unitsAndExpressions:

Units & Expressions
===================

Variables in Flow360 can represent solver inputs (e.g., time step, flow speed), intermediate parameters (for relationships between quantities), or derived quantities for post-processing.

Units define the dimensional context for these quantities. Flow360 automatically checks dimensional compatibility, preventing invalid combinations (like adding pressure and temperature).

Both **variables** and **units** can be managed through the WebUI or Python API, and Flow360 supports fully unit-aware symbolic expressions.

.. note::
   All Flow360 mathematical expressions are *unit-aware*, meaning any operation involving mismatched dimensions will raise an error.

.. toctree::

   units
   UserDefinedExpressions
   ExpressionsLegacy

