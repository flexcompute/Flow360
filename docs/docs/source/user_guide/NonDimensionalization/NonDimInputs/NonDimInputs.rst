.. currentmodule:: flow360

.. _non_dim_input_userGuide:

Non-Dimensional Inputs
======================

.. note::
   Thanks to :ref:`Units <python_API_units_introduction>`, manual dimensionalization of inputs is rarely needed. In most cases, you can simply provide the dimensional value and the unit system will be inferred automatically.

Most input variables in the Flow360 Python API accept dimensional values and non-dimensionalization is automatically performed during the preprocessing. 
However, there are still input variables defined using C-style string expression that only accept non-dimensional values, such as :py:class:`AngleExpression`, :py:class:`HeatFlux`, etc.
A string expression cannot carry units and none are inferred for it, so the number you write is already the non-dimensional one.
In this section, we demonstrate how to compute the non-dimensional value for these input variables.

.. caution::
   Several of the fields in the table below accept either a dimensional value or a string expression, and only the string form is non-dimensional. :code:`Freestream(velocity=(29, 0, 0) * u.m / u.s)` is 29 m/s, while :code:`Freestream(velocity=("29", "0", "0"))` is 29 times :math:`U_\text{scale}` — for an :py:class:`AerospaceCondition` that is Mach 29. Only the string form can vary in space or time.

Theoretically, the reference values for non-dimensionalization can be arbitrary as long as the resulting equations are identical to the original ones, 
but in practice, the reference values are usually selected based on some typical parameters of problems and flow characteristics to avoid confusion.
The following table shows the usage of reference values to obtain the non-dimensional input variables:

.. _tab_non_dim_input:
.. csv-table:: Usage of reference values for non-dimensional inputs in Flow360
   :file: Tables/nondim_input.csv
   :widths: 10,10,60
   :header-rows: 1
   :delim: @

.. note::

   1. All reference values can be accessed via the Python API as shown in the :ref:`Reference Quantities Table <table_ref_value>`.
   2. Many input variables are non-dimensionalized with the :ref:`reference velocity scaling <reference_velocity_scaling>` :math:`U_\text{scale}` (see :ref:`Introduction <non_dim_intro>`). In the examples below, we assume :math:`U_\text{scale}` has been computed as shown there.

.. _non_dim_omega:

Example: Convert RPM to non-dimensional rotating speed :code:`omega`
--------------------------------------------------------------------

The RPM determines the angular speed, from it we can calculate the non-dimensional :code:`omega` used in defining the :py:class:`AngleExpression`.

.. math::
    \text{omega} = \frac{\omega}{U_\text{scale}/L_\text{gridUnit}}.

where :math:`U_\text{scale}` is the :ref:`reference velocity scaling <reference_velocity_scaling>`.

Assume the RPM = 800, the non-dimensional :code:`omega_radians` value then becomes:

.. code-block:: python
   :linenos:

   omega = 800 * fl.u.rpm / (U_scale / project.length_unit)
   omega_radians = omega.to(fl.u.rad).value


.. _non_dim_volumetric_heat_source:

Example: Compute non-dimensional :py:attr:`~Solid.volumetric_heat_source`
----------------------------------------------------------------------------------------------------

For conjugate heat transfer simulations, the non-dimensional heat sources (:py:attr:`~Solid.volumetric_heat_source`) of a solid zone are found from the dimensional :math:`Q_s` as:

.. math::

   \text{volumetricHeatSource}= \frac{Q_s}{Q_{ref}} = \frac{Q_s L_{gridUnit}}{\rho_{\infty} U_\text{scale}^3}

where :math:`U_\text{scale}` is the :ref:`reference velocity scaling <reference_velocity_scaling>`.

Assume the :math:`Q_s=10\;\text{W}/\text{m}^3`, the non-dimensional :code:`volumetric_heat_source` value can be obtained as:

.. code-block:: python
   :linenos:

   density = operating_condition.thermal_state.density
   Q_s = 10 * fl.u.W / fl.u.m**3
   volumetric_heat_source = (Q_s * project.length_unit / (density * U_scale ** 3)).to("dimensionless").value

.. _non_dim_heat_flux:

Example: Compute non-dimensional :py:class:`HeatFlux`
----------------------------------------------------------------------------------------------------
The non-dimensional heat flux for a wall boundary condition can be calculated by dividing the dimensional heat flux :math:`q` by the reference value:

.. note::

   **Sign convention.** A positive heat flux removes energy from the fluid (heat flows from the fluid into the wall, cooling the fluid), while a negative heat flux adds energy to the fluid (heat flows from the wall into the fluid, heating it). See the :doc:`wall boundary condition </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/01.wall>` page.

.. math::

   \text{heatFlux} = \frac{q}{q_{ref}} = \frac{q}{\rho_{\infty} U_\text{scale}^3}

where :math:`U_\text{scale}` is the :ref:`reference velocity scaling <reference_velocity_scaling>`.

Assume the :math:`q=10\;\text{W}/\text{m}^2`, the non-dimensional :code:`heat_flux` value can be obtained as:

.. code-block:: python
   :linenos:

   density = operating_condition.thermal_state.density
   q = 10 * fl.u.W / fl.u.m**2
   heat_flux = (q / (density * U_scale ** 3)).to("dimensionless").value

.. _non_dim_boundary_velocity:

Example: Compute a non-dimensional spatially varying boundary velocity
----------------------------------------------------------------------------------------------------

:py:attr:`Wall.velocity` and :py:attr:`Freestream.velocity` accept either a dimensional value or a per-component
C-style string expression, and only the string form can vary in space or time. The string form is non-dimensional:
each component is divided by :math:`U_\text{scale}`, and the coordinates :code:`x`, :code:`y`, :code:`z` inside it
are divided by :math:`L_{gridUnit}`.

Consider an atmospheric boundary layer profile, a log law with a friction velocity :math:`u_* = 1.2\;\text{m}/\text{s}`,
a roughness length :math:`z_0 = 0.03\;\text{m}` and von Karman constant :math:`\kappa = 0.41`, capped at
:math:`29\;\text{m}/\text{s}`, on a mesh built in metres:

.. math::

   u(z) = \min\left(\frac{u_*}{\kappa}\ln\frac{z + z_0}{z_0},\; 29\;\text{m}/\text{s}\right)

Both the coefficient and the cap have to be divided by :math:`U_\text{scale}`, and :math:`z_0` by :math:`L_{gridUnit}`:

.. code-block:: python
   :linenos:

   u_star, z_0, kappa = 1.2 * fl.u.m / fl.u.s, 0.03 * fl.u.m, 0.41
   coefficient = (u_star / kappa / U_scale).value
   cap = (29 * fl.u.m / fl.u.s / U_scale).value
   z_0_nondim = (z_0 / project.length_unit).value

   velocity = (
       f"min({coefficient} * log((max(0.0, z) + {z_0_nondim}) / {z_0_nondim}), {cap})",
       "0",
       "0",
   )

.. caution::

   Writing the dimensional numbers straight into the string is a silent error, not a validation failure. With the
   cap left as :code:`1.0` the profile tops out at one :math:`U_\text{scale}` — for an :py:class:`AerospaceCondition`
   that is Mach 1, roughly 340 m/s rather than the intended 29 m/s. The case still converges and the contours still
   look plausible.

.. _alphaBetaAngles:

Define the angle of attack :code:`alpha` and sideslip angle :code:`beta`
------------------------------------------------------------------------

According to Flow360's definitions of the angle of attack :math:`\alpha` and the sideslip angle :math:`\beta`, with respect to the grid coordinates, the following values of velocity components are imposed at a :code:`Freestream` farfield boundary:

.. math::
      :label: EQ_FreestreamBC

      U_{\infty} &= \text{Mach} \cdot cos(\beta) \cdot cos(\alpha) \\
      V_{\infty} &= - \text{Mach} \cdot sin(\beta) \\
      W_{\infty} &= \text{Mach} \cdot cos(\beta) \cdot sin(\alpha)

where, the velocity components are nondimensionalized by the :ref:`reference velocity scaling <reference_velocity_scaling>` :math:`U_\text{scale}`. The effects of these two angles are used to compute the forces in stability axes rather than body axes, :math:`CL` and :math:`CD`, as follows: 

.. math::
      :label: EQ_CL_CD

      C_{L} &= C_{Fz}\cdot cos(\alpha) - C_{Fx}\cdot sin(\alpha)\\
      C_{D} &= C_{Fx} \cdot \cos(\alpha) \cos(\beta)
         - C_{Fy} \cdot \sin(\beta)
         + C_{Fz} \cdot \sin(\alpha) \cos(\beta)

Angle of attack :math:`\alpha` and sideslip angle :math:`\beta` can be expressed by readjusting the above equations in the following way:

.. math::
      :label: EQ_Alpha_Beta

      \alpha = tan^{-1} \left( \frac{W_\infty}{U_\infty} \right)\\
      \beta = sin^{-1} \left( -\frac{V_\infty}{\text{Mach}} \right)
