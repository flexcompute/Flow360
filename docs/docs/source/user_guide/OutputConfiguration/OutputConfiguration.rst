.. _outputConfiguration:

********************
Output Configuration
********************

Outputs control what data Flow360 writes during and after a simulation. You can export flow field data on volumes, surfaces, slices, isosurfaces, and at probe points—each with configurable fields and save frequency. Additionally, you can configure force and moment outputs, including total forces on specific models and custom force distributions along arbitrary directions. Results are saved in ParaView (``.vtu``/``.vtp``), Tecplot (``.szplt``), or CSV formats.

.. important::
   Outputs must be configured **before** running the simulation. Data that was not requested in the output configuration cannot be retrieved after the simulation completes. You would need to re-run the case with the desired outputs enabled.

.. _availableOutputs_userGuide:

Available Outputs 
=================
The table below lists all available output types. It also contains links to the documentation of each type for both the GUI and the Python API interfaces. 
The `File` column shows the path to the file that contains the data produced by each output for custom post-processing. All of the files can be found in the :doc:`assets menu </gui_guide/01.introduction/03.workbench-layout/04.top-bar>` in the WebUI.
The `API Access` column shows the property on the :class:`~flow360.component.case.Case` object used to access each result programmatically (see :ref:`python_api_download_results` for how to retrieve a case from cloud).


.. csv-table:: Available Output Types
   :header: "Output Type", "Data", "Availability", "File", "API Access"
   :widths: 18, 25, 10, 27, 20
   :file: ./Tables/output_list.csv


.. _multipleSurfaceOutputs_userGuide:

Multiple Outputs for the Same Surface
=====================================

The same surface can be assigned to more than one surface output. This is useful for unsteady and aeroacoustic simulations, where the same surface is often needed at more than one cadence or in more than one format. A typical pattern pairs a detailed output (which saves a full set of fields at a high frequency for flow-field and noise-source analysis, kept on the cloud and downloaded only when needed because of its size) with a lightweight output (which saves a compact set of fields less often for quick download and comparison in third-party tools such as ParaView or Tecplot).

When several surface outputs share a surface, each one must be given a unique name. The name is appended as a suffix to that output's filenames, so the data from each output is written to separate files and never overwrites the others.

.. note::
   The uniqueness requirement applies only between outputs of the same type. An instantaneous surface output and a time-averaged surface output may share a surface without unique names, because they already write to separate files. When several outputs of the same type are combined into a single file, their names must still resolve to distinct suffixes.

To configure this in each interface, see the :doc:`Surface Output </gui_guide/02.simulation-setup/04.output/02.outputs-list/03.surface-output>` page of the WebUI guide and the :doc:`output configuration </python_api/API_reference/outputs/output_index>` section of the Python API reference.

.. _outputFields_userGuide:

Output Fields
=============

.. note::
   The fields listed below are the **default output fields** provided by Flow360. All values are **non-dimensional** unless otherwise noted. See :ref:`Non-Dimensional Outputs <non_dim_output_userGuide>` for dimensionalization formulas.

.. important::
   **Custom and Dimensional Outputs**: Use :ref:`User Variables <UserVariable>` to define custom output expressions or to output existing fields in dimensional units (e.g., ``velocity_m_per_s``, ``pressure_pa``, ``wall_shear_stress_magnitude_pa``). See :ref:`Units & Expressions <unitsAndExpressions>` for details.


Universal Fields
----------------

Available for all output types (Volume, Surface, Slice, Isosurface, Probe):

.. csv-table::
   :header-rows: 1
   :widths: 50, 50
   :file: ./Tables/universal_fields.csv


Surface-Specific Fields
-----------------------

Available only for Surface Output and Surface Probe Output:

.. csv-table::
   :header-rows: 1
   :widths: 50, 50
   :file: ./Tables/surface_fields.csv


Volume and Slice-Specific Fields
--------------------------------

Available only for Volume Output and Slice Output:

.. csv-table::
   :header-rows: 1
   :widths: 50, 50
   :file: ./Tables/volume_fields.csv

.. _bet_metrics_output_variables:

BET Metrics Output Variables
-----------------------------

The ``betMetrics`` and ``betMetricsPerDisk`` output fields provide Blade Element Theory (BET) metrics for analyzing rotor and propeller performance. These fields are available when using BET models in volume zones. The ``betMetrics`` field includes data from all BET disks with possible overlapping, while ``betMetricsPerDisk`` provides separate outputs for each disk to avoid overlap.

The following variables are included in the betMetrics output:

1. **VelocityRelative**: Relative velocity with respect to the rotating reference frame (non-dimensional).

2. **AlphaRadians**: Local angle of attack in radians.

3. **CfAxial**: Axial aerodynamic force coefficient.

4. **CfCircumferential**: Circumferential aerodynamic force coefficient.

5. **TipLossFactor**: Factor to model the effect of blade tip.

6. **LocalSolidityIntegralWeight**: Local solidity multiplied by the integral weight.

.. note::
   Detailed explanations of these variables, including their mathematical formulations, are available in the BET disk section of the Formulations documentation.


Hybrid RANS-LES Model Outputs
------------------------------

The ``SpalartAllmaras_hybridModel`` and ``kOmegaSST_hybridModel`` output fields provide diagnostic variables for hybrid RANS-LES simulations (DDES and ZDES). These fields are available for **Volume Output** and **Slice Output** only.

.. important::
   **Requirements**: 
   
   - ``SpalartAllmaras_hybridModel`` can only be specified when using the **Spalart-Allmaras** turbulence model with hybrid RANS-LES enabled (DDES or ZDES).
   - ``kOmegaSST_hybridModel`` can only be specified when using the **kOmegaSST** turbulence model with hybrid RANS-LES enabled (DDES or ZDES).
   - Hybrid models require unsteady simulations—they are not available for steady-state cases.

The specific variables included in each hybrid model output depend on the shielding function used:

DDES Variables
^^^^^^^^^^^^^^

When ``shielding_function="DDES"``, the hybrid model output includes five variables:

1. **f_d**: The shielding function that delineates RANS and LES regions. When ``f_d`` = 0, RANS is fully applied; when ``f_d`` = 1, LES is used. Intermediate values represent a smooth transition between regimes.

2. **r_d**: A modified ratio of the modeled length scale to the wall distance, from which ``f_d`` is derived.

3. **DDES_lengthRANS**: The wall distance from the computational cell to the nearest solid boundary.

4. **DDES_lengthScale**: The characteristic DES length scale: :math:`\tilde{d} \equiv d - f_d \max(0, d - C_{DES}*\Delta)`

5. **DDES_lengthLES**: The characteristic LES length scale: :math:`C_{DES}*\Delta`

Among these variables, ``f_d`` is the most significant for identifying and visualizing regions dominated by RANS vs. LES behavior.

ZDES Variables
^^^^^^^^^^^^^^

When ``shielding_function="ZDES"``, the hybrid model output includes four variables:

1. **ZDES_fp**: The enhanced shielding function that determines whether RANS or LES is used. When ``ZDES_fp`` = 0, RANS is active; when ``ZDES_fp`` = 1, LES is active. This function is computed from ``ZDES_fd``, ``ZDES_fR``, and ``ZDES_fp2``.

2. **ZDES_fd**: Original DDES shielding function used in computing ``ZDES_fp``.

3. **ZDES_fR**: Component that disables or inhibits the secondary shielding function in regions where vorticity magnitude increases away from walls (designed to disable the secondary shielding where a shear layer is detected above a wall). Used in computing ``ZDES_fp``.

4. **ZDES_fp2**: Causes the model to revert to RANS mode in the outer portion of boundary layers. Used in computing ``ZDES_fp``.

.. _bet_coefficient_distributions:

BET Coefficient Distributions
=============================

For any case that includes one or more :ref:`BET disks <BET_Translators>`, Flow360 automatically writes tabular (CSV) summaries of the blade-element loading alongside the field outputs. These tables let you analyze rotor and propeller performance directly. They are produced whenever a BET model is present, so no additional output configuration is required.

All coefficients in these tables are **non-dimensional** (see :ref:`Non-Dimensional Outputs <non_dim_output_userGuide>`). Force coefficients are normalized by :math:`q_\infty \, S_{ref}` and moment coefficients by :math:`q_\infty \, S_{ref} \, L_{ref}`, where :math:`q_\infty` is the freestream dynamic pressure and :math:`S_{ref}`, :math:`L_{ref}` are the reference area and reference length.

Per-disk force and moment coefficients
--------------------------------------

The file ``bet_force_coefficients_v2.csv`` contains the integrated force and moment coefficients of each BET disk at every recorded step. The first two columns are ``physical_step`` and ``pseudo_step``; the remaining columns repeat for each disk ``i``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Quantity
   * - ``Disk<i>_CFx``, ``Disk<i>_CFy``, ``Disk<i>_CFz``
     - Force coefficient components along the global x, y, z axes.
   * - ``Disk<i>_CMx``, ``Disk<i>_CMy``, ``Disk<i>_CMz``
     - Moment coefficient components about the global x, y, z axes, taken about the case moment center.
   * - ``Disk<i>_CL``
     - Lift coefficient (force projected onto the lift direction).
   * - ``Disk<i>_CD``
     - Drag coefficient (force projected onto the freestream/drag direction).

By default each disk is identified by its global index (``Disk<i>``). When the table is retrieved through the Python API the column headers can be renamed to use the BET model and cylinder names defined in the simulation (for example ``Disk0`` becomes ``<BETName>_<CylinderName>``).

Sectional (radial) distribution
--------------------------------

The file ``bet_forces_radial_distribution_v2.csv`` contains the spanwise loading distribution. Each row corresponds to one radial station (loading node) along the blade, so the number of rows matches the number of loading nodes used to resolve the disk (the ``n_loading_nodes`` value from the :ref:`BET disk setup <BET_Translators>`). For each disk ``i`` the columns are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Column
     - Quantity
   * - ``Disk<i>_All_Radius``
     - Radial position of the station.
   * - ``Disk<i>_Blade<b>_All_ThrustCoeff``
     - Sectional thrust coefficient at that radius for blade ``b``.
   * - ``Disk<i>_Blade<b>_All_TorqueCoeff``
     - Sectional torque coefficient at that radius for blade ``b``.

Because the thrust and torque coefficients are reported separately for each blade, the table resolves the radial (spanwise) variation of the loading and, for time-accurate :ref:`BET Line <bet_disk_knowledge_base>` simulations where the blades occupy distinct azimuthal positions, the per-blade (azimuthal) variation as well.

Accessing the tables
--------------------

Both files are written to the case output and can be downloaded from the :doc:`assets menu </gui_guide/01.introduction/03.workbench-layout/04.top-bar>` in the WebUI, together with the other result files. They can also be retrieved and processed programmatically through the ``case.results`` interface (``case.results.bet_forces`` and ``case.results.bet_forces_radial_distribution``). See :ref:`Download Results <python_api_download_results>` and the :doc:`Results API reference </python_api/API_reference/results>` for the available models and methods.

.. seealso::
   
   - :ref:`Outputs API Reference <outputs>` — Python API classes for all output types
   - :ref:`Non-Dimensional Outputs <non_dim_output_userGuide>` — formulas to convert non-dimensional values to physical units
   - :ref:`Converting to Physical Units (N, N·m) <force_moment_physical_conversion>` — step-by-step conversion of force/moment coefficients and raw BET/AD/PM outputs to Newtons and Newton-meters
   - :ref:`User Variables <UserVariable>` — define custom expressions and dimensional outputs
   - :ref:`Units & Expressions <unitsAndExpressions>` — unit-aware variable system
   - :ref:`Results Processing <resultsProcessing_userGuide_workflowsInterfaces>` — processing the results after the simulation
   - :ref:`BET Coefficient Distributions <bet_coefficient_distributions>` (tabular BET force, moment and sectional-loading exports)

