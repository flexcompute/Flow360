.. _non_dim_coeff_userGuide:

.. currentmodule:: flow360

Force and Moment Coefficients
==============================

Flow360 exports standard aerodynamic coefficients (:math:`C_L`, :math:`C_D`, :math:`C_F`, :math:`C_M`) alongside the :ref:`non-dimensional output fields <non_dim_output_userGuide>`. These are available in ``total_forces_v2.csv``, ``surface_forces_v2.csv``, and ``force_output_*_v2.csv``, or via :code:`case.results.total_forces` / :code:`case.results.surface_forces`.

.. attention::

   Force and moment coefficients use :math:`U_\text{ref}` (``case.params.reference_velocity``), **not** the :ref:`reference velocity scaling <reference_velocity_scaling>` :math:`U_\text{scale}` used for other solver fields.

   BET Disk, Actuator Disk, and Porous Media report raw forces and moments (not coefficients) that use :math:`U_\text{scale}` — see :ref:`Raw solver output conversion <conversion_raw_to_si>` or the :ref:`Outputs page <non_dim_output_userGuide>` for their formulas.

Coefficient Definitions
-----------------------

.. _tab_non_dim_coeff:
.. csv-table:: Force and moment coefficient definitions
   :file: Tables/nonDimCoefficients.csv
   :widths: 10 50
   :header-rows: 1
   :delim: @

.. note::
   All reference values are listed in the :ref:`Reference Quantities Table <table_ref_value>`.

Output Conventions
------------------

The conventions assume z-up, y-spanwise (+ starboard), x-axial (+ freestream), as shown below.

.. _axis_conventions:
.. figure:: Figures/AxisConventions.png
   :align: center
   :scale: 80%

   Axis conventions (CRM geometry)

.. _tab_ForcesMoments:
.. csv-table:: Force and moment output conventions
   :file: Tables/force_moments.csv
   :widths: 20 60
   :align: center
   :header-rows: 1
   :delim: @


.. _force_moment_physical_conversion:

Converting to Physical Units (N, N·m)
--------------------------------------

Two conversion paths exist depending on the data source. This section is a self-contained reference for both.

.. _conversion_dynamic_pressure:

Dynamic pressure
^^^^^^^^^^^^^^^^^^^^^^^^^^

All coefficient-based conversions start from the dynamic pressure built with :math:`U_\text{ref}` (see :ref:`table_ref_value` for API access):

.. math::

   q_\text{ref} = \tfrac{1}{2}\,\rho_\infty\,U_\text{ref}^2 \qquad [\text{Pa}]

.. code-block:: python

   density = case.params.operating_condition.thermal_state.density
   U_ref   = case.params.reference_velocity
   A_ref   = case.params.reference_geometry.area
   L_ref   = case.params.reference_geometry.moment_length   # 3-vector

   q_ref = 0.5 * density * U_ref**2


.. _conversion_coefficients_to_si:

Coefficient → Newtons / Newton-meters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies to ``surface_forces_v2.csv``, ``total_forces_v2.csv``, ``force_output_*_v2.csv``, and the postprocessed coefficient files (``bet_force_coefficients_v2.csv``, ``actuatorDisk_force_coefficients_v2.csv``, ``porous_media_force_coefficients_v2.csv``).

**Force:**

.. math::

   F\;[\text{N}] = C_F \cdot q_\text{ref} \cdot A_\text{ref}

**Moment** (per component — :math:`L_\text{ref}` can differ per axis):

.. math::

   M_i\;[\text{N·m}] = C_{M_i} \cdot q_\text{ref} \cdot A_\text{ref} \cdot L_{\text{ref},\,i}

.. code-block:: python

   force_scale  = q_ref * A_ref                  # N per unit C_F
   moment_scale = q_ref * A_ref * L_ref           # N·m per unit C_M (3-vector)

   Lift            = CL  * force_scale             # N
   Pitching_Moment = CMy * moment_scale[1]         # N·m

.. tip::

   When ``q_ref``, ``A_ref``, and ``L_ref`` are retrieved through the Python API they already carry SI units, so you can call ``.to()`` directly to get the result in any target unit:

   .. code-block:: python

      Lift    = (CL  * force_scale).to('N')
      Moment  = (CMy * moment_scale[1]).to('N*m')

   See :ref:`python_api_calculate_dimensional_forces` for a complete worked example.


.. _conversion_raw_to_si:

Converting Flow360 Force Outputs to SI (BET / Actuator Disk / Porous Media)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solver writes **forces and moments in Flow360 non-dimensional units** for volume models in ``bet_forces_v2.csv``, ``actuatorDisk_output_v2.csv``, and ``porous_media_output_v2.csv``. These are **force outputs, not aerodynamic coefficients**: because they are dimensionalized using the solver's internal scaling — :ref:`reference velocity scaling <reference_velocity_scaling>` :math:`U_\text{scale}` and :math:`L_\text{gridUnit}` — rather than the aerodynamic reference quantities :math:`U_\text{ref}` and :math:`A_\text{ref}`, a dedicated conversion is required to recover SI forces and moments.

.. math::

   k_F &= \rho_\infty\,U_\text{scale}^2\,L_\text{gridUnit}^2  \\
   k_M &= \rho_\infty\,U_\text{scale}^2\,L_\text{gridUnit}^3 \\
   k_P &= \rho_\infty\,U_\text{scale}^3\,L_\text{gridUnit}^2

.. math::

   F\;[\text{N}]   &= F_\text{raw} \times k_F \\
   M\;[\text{N·m}] &= M_\text{raw} \times k_M \\
   P\;[\text{W}]   &= P_\text{raw} \times k_P

.. code-block:: python

   L_grid = project.length_unit

   k_F = density * U_scale**2 * L_grid**2   # N   (force scale)
   k_M = density * U_scale**2 * L_grid**3   # N·m (moment scale)
   k_P = density * U_scale**3 * L_grid**2   # W   (power scale)

   force_x  = raw_force_x  * k_F   # N
   moment_x = raw_moment_x * k_M   # N·m

.. attention::

   **Do not mix the two paths.** Coefficient files use :math:`U_\text{ref}`, :math:`A_\text{ref}`, :math:`L_\text{ref}`. Raw BET/AD/PM files use :math:`U_\text{scale}` and :math:`L_\text{gridUnit}` (scaling factors :math:`k_F`, :math:`k_M`, :math:`k_P`).


.. _conversion_summary_table:

Summary
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Data source
     - CSV files
     - Conversion
   * - Surface / total coefficients
     - ``surface_forces_v2.csv``, ``total_forces_v2.csv``, ``force_output_*_v2.csv``
     - :math:`F = C_F \cdot q_\text{ref} \cdot A_\text{ref}`
   * - BET/AD/PM coefficients
     - ``bet_force_coefficients_v2.csv``, ``actuatorDisk_force_coefficients_v2.csv``, ``porous_media_force_coefficients_v2.csv``
     - Same as above
   * - BET/AD/PM raw forces
     - ``bet_forces_v2.csv``, ``actuatorDisk_output_v2.csv``, ``porous_media_output_v2.csv``
     - :math:`F = F_\text{raw} \times k_F`
   * - Moment coefficients
     - All coefficient files
     - :math:`M_i = C_{M_i} \cdot q_\text{ref} \cdot A_\text{ref} \cdot L_{\text{ref},i}`
   * - BET/AD/PM raw moments
     - Raw files above
     - :math:`M = M_\text{raw} \times k_M`


.. seealso::

   - :ref:`python_api_calculate_dimensional_forces` — Complete Python snippet for coefficient → N, N·m conversion.
   - :ref:`betDiskLoadingNote` — BET raw output definitions and worked examples.
   - :ref:`ADoutput` — Actuator Disk raw output definitions.
