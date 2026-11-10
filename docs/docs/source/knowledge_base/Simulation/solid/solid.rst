.. _knowledge_base_solid:

.. currentmodule:: flow360

Solid
=====

:py:attr:`~Solid.material`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solid material is required for the heat transfer volume zone, which can be defined using :class:`SolidMaterial`. For example, the aluminum can be defined as:

.. code-block:: python

    aluminum = fl.SolidMaterial(
        name="aluminum",
        thermal_conductivity=235 * u.kg / u.s**3 * u.m / u.K,
        density=2710 * u.kg / u.m**3,
        specific_heat_capacity=903 * u.m**2 / u.s**2 / u.K,
    )

:py:attr:`~Solid.initial_condition`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the unsteady case, the initial condition of the heat transfer volume zone can be defined using :class:`HeatEquationInitialCondition`

.. code-block:: python

    initial_condition = fl.HeatEquationInitialCondition(temperature="1.0")
