.. _python_API_bet_migration:

.. currentmodule:: flow360

BET Migration Tools
===================

Overview
--------

The Blade Element Theory (BET) migration tools in Flow360 provide a robust framework for converting legacy V1 BET configurations to the current Flow360 version. This guide outlines the process of migrating BET disk configurations to the latest Flow360 format.

Key Components
--------------

The migration tools consist of two primary functions:

.. autofunction:: flow360.migration.BETDisk.read_single_v1_BETDisk
.. autofunction:: flow360.migration.BETDisk.read_all_v1_BETDisks

These functions:

- Automatically assign units to legacy BETDisk parameters based on the mesh unit.
- Convert legacy parameter names to the current API format.
- Create the corresponding Cylinder entity required for each BET disk.
- Calculate the non-dimensional rotation speed (omega) using freestream conditions.

Usage Examples
--------------

Single BET Disk Migration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import flow360 as fl
    from flow360.component.simulation.migration import BETDisk

    # Read and convert single BET disk configuration
    my_BETDisk = BETDisk.read_single_v1_BETDisk(
        file_path="./BET_tutorial_Flow360.json",
        mesh_unit=fl.u.m,
        freestream_temperature=300 * fl.u.K,
    )

    # Use the converted BET disk in simulation
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            models=[my_BETDisk],
            # ... other parameters ...
        )

    # Modify BET disk properties if needed
    params.models[0].omega = 500 * fl.u.rpm

Multiple BET Disks Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import flow360 as fl
    from flow360.component.simulation.migration import BETDisk

    # Initialize simulation parameters
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            operating_condition=fl.AerospaceCondition(velocity_magnitude=10)
        )

    # Read and convert multiple BET disk configurations
    bet_disks = BETDisk.read_all_v1_BETDisks(
        file_path="./multi_BET_tutorial_Flow360.json",
        mesh_unit=fl.u.m,
        freestream_temperature=params.operating_condition.thermal_state.temperature,
    )

    # Use the converted BET disks in simulation
    params = fl.SimulationParams(
        models=bet_disks,
        # ... other parameters ...
    )

    # Modify properties of specific BET disks if needed
    params.models[0].omega = 500 * fl.u.rpm  # First disk
    params.models[1].omega = 750 * fl.u.rpm  # Second disk

Parameter Mapping
-----------------

The migration process automatically handles the conversion of legacy parameters to their current equivalents. The following tables show the complete mapping of parameters:

BET Disk Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 30
   :header-rows: 1

   * - Legacy Parameter
     - Current Parameter
   * - ``rotationDirectionRule``
     - :py:attr:`~BETDisk.rotation_direction_rule`
   * - ``numberOfBlades``
     - :py:attr:`~BETDisk.number_of_blades`
   * - ``ReynoldsNumbers``
     - :py:attr:`~BETDisk.reynolds_numbers`
   * - ``chord_ref``
     - :py:attr:`~BETDisk.reynolds_numbers`
   * - ``n_loading_nodes``
     - :py:attr:`~BETDisk.n_loading_nodes`
   * - ``sectional_radiuses``
     - :py:attr:`~BETDisk.sectional_radiuses`
   * - ``sectional_polars``
     - :py:attr:`~BETDisk.sectional_polars`
   * - ``mach_numbers``
     - :py:attr:`~BETDisk.mach_numbers`
   * - ``lift_coeffs``
     - :py:attr:`~BETDisk.lift_coeffs`
   * - ``drag_coeffs``
     - :py:attr:`~BETDisk.drag_coeffs`
   * - ``tip_gap``
     - :py:attr:`~BETDisk.tip_gap`
   * - ``initial_blade_direction``
     - :py:attr:`~BETDisk.initial_blade_direction`
   * - ``blade_line_chord``
     - :py:attr:`~BETDisk.blade_line_chord`

Cylinder Entity Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parameters are used to create the associated cylinder entity for the BET disk:

.. list-table::
   :widths: 30 30
   :header-rows: 1

   * - Legacy Parameter
     - Current Parameter
   * - ``axisOfRotation``
     - :py:attr:`~Cylinder.axis`
   * - ``centerOfRotation``
     - :py:attr:`~Cylinder.center`
   * - ``radius``
     - :py:attr:`~Cylinder.outer_radius`
   * - ``thickness``
     - :py:attr:`~Cylinder.height`

Note: The cylinder entity is automatically created with:

- :py:attr:`~Cylinder.inner_radius` set to 0
- :py:attr:`~Cylinder.name` generated as "bet_cylinder{N}" where N is the disk index

Important Considerations
------------------------

1. **Units**: The migration process requires specification of the mesh unit and freestream temperature for proper unit assignment.

2. **Geometry Parameters**: The tool automatically converts geometric parameters such as:

   - Axis of rotation
   - Center of rotation
   - Radius
   - Thickness

3. **Aerodynamic Data**: The migration preserves:

   - Lift coefficients
   - Drag coefficients
   - Mach numbers
   - Reynolds numbers

4. **Tip Gap Handling**: The tool supports both finite and infinite tip gap configurations.

Best Practices
--------------

1. Always verify the converted parameters before running simulations
2. Ensure proper mesh resolution around BET disk regions
3. Consider tip gap effects in the simulation setup

Example Configuration
---------------------

A typical Legacy BET disk configuration includes:

.. code-block::

    {
        "BETDisks": [
            {
                "axisOfRotation": [0, 0, 1],
                "centerOfRotation": [0, 0, 0],
                "radius": 1.0,
                "thickness": 0.1,
                "numberOfBlades": 4,
                "omega": 1000,
                "sectionalPolars": [...],
                "twists": [...],
                "chords": [...]
            }
        ]
    }
