.. _python_API_units_introduction:

Units
=====

Most of the Python API inputs are required to be dimensional. User can provide the units per input or use a :ref:`unit system <unit_system>` to let it automatically handle unit inference.

.. note::
    There are exceptions where the input is expected to be non-dimensional. These are most notably the string expressions (for example :py:attr:`~flow360.Fluid.initial_condition`) since they are directly executed inside the solver where everything is non-dimensional. When composing expressions user should bear in mind that the input and output variables are all non-dimensional.

For example to set the physical time step as 0.01 seconds, user can provide the input as:

.. code-block:: python

    from flow360 import u
    my_time_stepping = fl.Unsteady(step_size=0.01 * u.s)

.. note::
    As mentioned in the previous note, expressions are non-dimensional or have hardcoded unit. For example the following expression indicates the initial condition for fluid density will be 1.5 times of the density of the fluid at its given :ref:`operating condition <operating_condition>`:

    .. code-block:: python

        fluid_model = fl.Fluid(initial_condition=fl.NavierStokesInitialCondition(rho="1.5"))

    Another example is when setting the rotation angle expression. It is assumed that the angle expression results in radians which can not be customized by the user:

    .. code-block:: python

        rotation_mode = fl.Rotation(
            volumes=fl.Cylinder(
                name="my_cyl",
                outer_radius=0.5 * u.cm,
                height=1 * u.km,
                axis=(0, 0, 1),
                center=(0, 0, 0)*u.mile,
            ),
            spec=fl.AngleExpression("5 * t"),
        )

    This means that the rotation angle (in radians) will be 5 times the non-dimensional time step.

For a full list of usable units, please refer to the documentation page of `unyt package <https://unyt.readthedocs.io/en/stable/unit_listing.html>`_.

To see how Flow360 performs non-dimensionalization, please refer to the :ref:`Nondimensional Inputs <non_dim_input_userGuide>` in our documentation.

To avoid obfuscation, some of the compound units are listed below:

.. _tab1_API_units:
.. table:: Units used in Flow360 Python API
    :align: center

    +-----------------------------------+----------------------------------+------------------+
    | Unit name                         | Dimension                        | Example unit     |
    +===================================+==================================+==================+
    | :code:`PressureType`              | :math:`M^{1} L^{-1} T^{-2}`      | u.Pa             |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`ViscosityType`             | :math:`M^{1} L^{-1} T^{-1}`      | u.Pa*u.s         |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`PowerType`                 | :math:`M^{1} L^{2} T^{-3}`       | u.W              |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`MomentType`                | :math:`M^{1} L^{2} T^{-2}`       | u.N*u.m          |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`HeatFluxType`              | :math:`M^{1} T^{-3}`             | u.W / u.m**2     |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`HeatSourceType`            | :math:`M^{1} L^{-1} T^{-3}`      | u.W / u.m**3     |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`SpecificHeatCapacityType`  | :math:`L^{2} T^{-2} Θ^{-1}`      | u.J / (u.kg*u.K) |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`ThermalConductivityType`   | :math:`M^{1} L^{1} T^{-3} Θ^{-1}`| u.W / (u.m*u.K)  |
    +-----------------------------------+----------------------------------+------------------+
    | :code:`MassFlowRateType`          | :math:`M^{1} T^{-1}`             | u.kg / u.s       |
    +-----------------------------------+----------------------------------+------------------+

.. _unit_system:

Unit System
-----------

Unit system provides convenience to the user to handle unit inference. User can define a unit system context and Flow360 API can automatically assign units to any input that lacks an explicit unit specification.

For example the following two code snippets are equivalent:

.. code-block:: python

    from flow360 import u
    
    meshing_defaults = fl.MeshingDefaults(
        boundary_layer_first_layer_thickness=0.0001 * u.m,
        surface_max_edge_length=0.0001 * u.m,
        curvature_resolution_angle=30 * u.deg,
    )

.. code-block:: python

    from flow360 import u
    
    with fl.SI_unit_system:
        meshing_defaults = fl.MeshingDefaults(
            boundary_layer_first_layer_thickness=0.0001 * u.m,
            surface_max_edge_length=0.0001,
            curvature_resolution_angle=30 * u.deg,
        )

Notice how in the second example we did not specify any unit for :code:`surface_max_edge_length` but since we constructed the model
under the :code:`SI_unit_system` context, Flow360 API automatically inferred the unit for :code:`surface_max_edge_length` as meter.

.. warning::
    Even though all unit systems provide default unit for angle, we will **NOT** add units to angle type inputs. This is because the default for angle type is not deemed universal and may vary depending on the user's conception.

    For example, the following code snippet will not work because we are missing unit for :code:`curvature_resolution_angle`:

    .. code-block:: python

        from flow360 import u
        
        with fl.SI_unit_system:
            meshing_defaults = fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=0.0001 * u.m,
                surface_max_edge_length=0.0001,
                curvature_resolution_angle=30,
            )
    
The default units for each unit system are as follows:

.. _tab2_API_unit_system:

==============================================================
Physical Dimensions and Unit Systems
==============================================================

This table lists the base and derived physical quantities expressed
in three major unit systems — **SI**, **CGS**, and **Imperial**.

.. list-table:: Unit systems available in Flow360 Python API
   :header-rows: 1
   :widths: 25 25 25 25
   :align: center
   :class: table-bordered table-striped

   * - **Dimension**
     - **SI**
     - **CGS**
     - **Imperial**
   * - Mass
     - 1.0 kg
     - 1.0 g
     - 1.0 lb
   * - Length
     - 1.0 m
     - 1.0 cm
     - 1.0 ft
   * - Angle
     - 1.0 rad
     - 1.0 rad
     - 1.0 rad
   * - Time
     - 1.0 s
     - 1.0 s
     - 1.0 s
   * - Temperature
     - 1.0 K
     - 1.0 K
     - 1.0 °F
   * - Velocity
     - 1.0 m/s
     - 1.0 cm/s
     - 1.0 ft/s
   * - Area
     - 1.0 m\ :sup:`2`
     - 1.0 cm\ :sup:`2`
     - 1.0 ft\ :sup:`2`
   * - Force
     - 1.0 N
     - 1.0 dyn
     - 1.0 lbf
   * - Pressure
     - 1.0 Pa
     - 1.0 dyn/cm\ :sup:`2`
     - 1.0 lbf/ft\ :sup:`2`
   * - Density
     - 1.0 kg/m\ :sup:`3`
     - 1.0 g/cm\ :sup:`3`
     - 1.0 lb/ft\ :sup:`3`
   * - Dynamic viscosity
     - 1.0 kg/(m·s)
     - 1.0 g/(cm·s)
     - 1.0 lb/(ft·s)
   * - Kinematic viscosity
     - 1.0 m\ :sup:`2`/s
     - 1.0 cm\ :sup:`2`/s
     - 1.0 ft\ :sup:`2`/s
   * - Power
     - 1.0 W
     - 1.0 erg/s
     - 1.0 hp
   * - Moment / Energy
     - 1.0 J
     - 1.0 erg
     - 1.0 ft·lbf
   * - Angular velocity
     - 1.0 rad/s
     - 1.0 rad/s
     - 1.0 rad/s
   * - Heat flux
     - 1.0 kg/s\ :sup:`3`
     - 1.0 g/s\ :sup:`3`
     - 1.0 lb/s\ :sup:`3`
   * - Volumetric heat source
     - 1.0 kg/(m·s\ :sup:`3`)
     - 1.0 g/(cm·s\ :sup:`3`)
     - 1.0 lb/(ft·s\ :sup:`3`)
   * - Specific heat capacity
     - 1.0 m\ :sup:`2`/(K·s\ :sup:`2`)
     - 1.0 cm\ :sup:`2`/(K·s\ :sup:`2`)
     - 1.0 ft\ :sup:`2`/(°F·s\ :sup:`2`)
   * - Thermal conductivity
     - 1.0 kg·m/(K·s\ :sup:`3`)
     - 1.0 cm·g/(K·s\ :sup:`3`)
     - 1.0 ft·lb/(°F·s\ :sup:`3`)
   * - Inverse area
     - 1.0 m\ :sup:`−2`
     - 1.0 cm\ :sup:`−2`
     - 1.0 ft\ :sup:`−2`
   * - Inverse length
     - 1.0 1/m
     - 1.0 1/cm
     - 1.0 1/ft
   * - Mass flow rate
     - 1.0 kg/s
     - 1.0 g/s
     - 1.0 lb/s
   * - Specific energy
     - 1.0 J/kg
     - 1.0 erg/g
     - 1.0 ft\ :sup:`2`/s\ :sup:`2`
   * - Frequency
     - 1.0 Hz
     - 1.0 1/s
     - 1.0 1/s
   * - Δ Temperature
     - 1.0 K
     - 1.0 K
     - 1.0 Δ°F



