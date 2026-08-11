.. _non_dim_intro:

.. currentmodule:: flow360


Introduction
*********************

Non-dimensionalization is essential for the Flow360 solver to reduce the number of variables and helps to provide better understanding of the underlying physics.
A non-dimensional variable is obtained by dividing its dimensional counterpart by an appropriately selected constant reference value.

The table below lists the reference quantities used throughout Flow360 and how to access them via the Python API. Here, :code:`project` is the cloud asset containing the :code:`case`, and :code:`params` is the :py:class:`SimulationParams` object (for a completed case: :code:`case.params`). See the :ref:`Python API guide <python_api>` for details.

.. _table_ref_value:

.. list-table:: Variables used in non-dimensionalization in Flow360
   :widths: 10 20 10 40
   :header-rows: 1

   * - Variable
     - Definition
     - SI unit
     - Python API access
   * - :math:`L_{gridUnit}`
     - Physical length per mesh unit (e.g. if the mesh is in feet, :math:`L_{gridUnit} = 0.3048\;\text{m}`)
     - m
     - .. code-block:: python

        project.length_unit
   * - :math:`C_\infty`
     - Speed of sound (freestream)
     - m/s
     - .. code-block:: python

        case.params.operating_condition \
        .thermal_state.speed_of_sound
   * - :math:`\rho_\infty`
     - Density (freestream)
     - kg/m³
     - .. code-block:: python

        case.params.operating_condition \
        .thermal_state.density
   * - :math:`\mu_\infty`
     - Dynamic viscosity (freestream)
     - N·s/m²
     - .. code-block:: python

        case.params.operating_condition \
        .thermal_state.dynamic_viscosity
   * - :math:`p_\infty`
     - Static pressure of freestream.

       .. note:: :math:`p_\infty` is used only for the pressure coefficient :math:`C_p`, not for general pressure non-dimensionalization (see :ref:`tab_non_dim_input`).
     - :math:`\text{N}/\text{m}^2`
     - .. code-block:: python

        case.params.operating_condition \
        .thermal_state.pressure
   * - :math:`T_\infty`
     - Temperature (freestream)
     - K
     - .. code-block:: python

        case.params.operating_condition \
        .thermal_state.temperature
   * - :math:`U_\text{ref}`
     - Reference velocity (used for force and moment coefficients)
     - m/s
     - .. code-block:: python

        case.params.reference_velocity
   * - :math:`A_\text{ref}`
     - Reference area
     - m²
     - .. code-block:: python

        case.params.reference_geometry \
        .area
   * - :math:`L_\text{ref}`
     - Moment length (can be a 3-component vector)
     - m
     - .. code-block:: python

        case.params.reference_geometry \
        .moment_length

.. _reference_velocity_scaling:

Reference Velocity Scaling
--------------------------

We introduce the **reference velocity scaling** :math:`U_\text{scale}`, which is the reference velocity used for velocity nondimensionalization in many Flow360 variables (such as velocity fields, heat flux, volumetric heat sources, and angular speeds). It also sets the pressure reference: pressure is non-dimensionalized by :math:`\rho_\infty U_\text{scale}^2`.

- :py:class:`AerospaceCondition`: :math:`U_\text{scale} = C_\infty` (speed of sound)
- :py:class:`LiquidOperatingCondition`: :math:`U_\text{scale}` is :py:attr:`~LiquidOperatingCondition.reference_velocity` (which corresponds to :py:attr:`~LiquidOperatingCondition.reference_velocity_magnitude` if set, otherwise :py:attr:`~LiquidOperatingCondition.velocity_magnitude`)

.. code-block:: python
   :name: code_dim_velocity

   if operating_condition.type_name == "LiquidOperatingCondition":
       U_scale = operating_condition.reference_velocity
   else:  # AerospaceCondition
       U_scale = operating_condition.thermal_state.speed_of_sound

.. note::
   It is important to distinguish :math:`U_\text{scale}` from :math:`U_\text{ref}` (see :ref:`table_ref_value`). :math:`U_\text{scale}` non-dimensionalizes solver fields: velocity by :math:`U_\text{scale}`, pressure by :math:`\rho_\infty U_\text{scale}^2`, heat flux by :math:`\rho_\infty U_\text{scale}^3`. :math:`U_\text{ref}` non-dimensionalizes **force and moment coefficients** (:math:`C_L`, :math:`C_D`, :math:`C_F`, :math:`C_M`). Mixing them up produces incorrect conversions.

Grid Unit Length
----------------

:math:`L_{gridUnit}` scales all length-related quantities. The mesh can use any unit; :ref:`tab_Lgridunit` lists common values.

.. _tab_Lgridunit:
.. list-table:: :math:`L_{gridUnit}` for common mesh units
   :widths: 15 35
   :header-rows: 1

   * - Mesh unit
     - :math:`L_{gridUnit}`
   * - metres [m]
     - 1.0 m
   * - millimetres [mm]
     - 0.001 m
   * - feet [ft]
     - 0.3048 m
   * - inches [in]
     - 0.0254 m
   * - unit chords [c = 1.0]
     - value of c_ref in metres

The following pages explain how these reference quantities apply to:

- :ref:`Inputs <non_dim_input_userGuide>` — non-dimensionalizing user-supplied values (angular speed, heat flux, …)
- :ref:`Outputs <non_dim_output_userGuide>` — converting solver fields back to physical units
- :ref:`Force & moment coefficients <non_dim_coeff_userGuide>` — converting :math:`C_L`, :math:`C_D`, :math:`C_M`, and raw BET/AD/PM forces to Newtons and Newton-meters
