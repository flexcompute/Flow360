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

.. _outputFrequency_userGuide:

Output Save Frequency
=====================

How often an output is written is set by ``Frequency``, and the point at which saving begins is set by ``Frequency offset``. Every output type that writes its data at intervals carries both settings, for example :py:attr:`~flow360.SurfaceOutput.frequency` and :py:attr:`~flow360.SurfaceOutput.frequency_offset` on a surface output. A frequency of ``-1`` writes the output at the end of the simulation only, which is the default. In the WebUI the two fields appear once **Save interval** is set to ``Custom``. The unit in which both are counted depends on the type of simulation.

Unsteady simulations
--------------------

Both values are counted in **physical time steps**, and can be set through either interface.

For child (forked) cases they refer to the **global** time step, which is inherited from the parent. A child of a case that finished at time step 174 starts at time step 175, so a frequency of 100 writes at global time steps 200 (25 time steps into the child run), 300, and so on.

Steady simulations (Python API only)
------------------------------------

The same two settings also apply to steady simulations, where they are counted in **pseudo steps**. Intermediate states of a converging steady run can therefore be saved rather than only the final one, which is useful for following how the solution develops. This is supported for volume (:class:`~flow360.VolumeOutput`), surface (:class:`~flow360.SurfaceOutput`), slice (:class:`~flow360.SliceOutput`), isosurface (:class:`~flow360.IsosurfaceOutput`) and render (:class:`~flow360.RenderOutput`) outputs.

Setting a save frequency on a steady case is currently possible through the Python API only. In the WebUI, **Save interval** is fixed at ``Save at end`` whenever the time stepping is steady, so the fields are not editable there. A frequency set through the Python API is preserved and shown when the case is opened in the WebUI.

For a steady child (forked) case the pseudo-step count restarts at 0, so both the frequency and the offset are counted from the start of the child run rather than continuing the count of the parent.

.. note::
   A converging steady run often needs only a few hundred pseudo steps, so a small frequency writes a large number of files and slows the case down. A validation warning is raised when the requested frequency would save an output many times over the course of the run, and recommends increasing the frequency.

.. note::
   Time-averaged outputs are supported in unsteady simulations only, so the pseudo-step counting above applies to their instantaneous counterparts rather than to the time-averaged outputs themselves.

To configure this in each interface, see the :doc:`outputs list </gui_guide/02.simulation-setup/04.output/02.outputs-list/README>` of the WebUI guide, which documents **Save interval**, ``Frequency`` and ``Frequency offset`` on each output type, and the :doc:`output configuration </python_api/API_reference/outputs/output_index>` section of the Python API reference.

.. _timeAverageStatistics_userGuide:

Time-Averaged Statistics
========================

A time-averaged output accumulates one or more statistics over its averaging window, selected with ``statistics`` (:py:attr:`~flow360.TimeAverageSurfaceOutput.statistics`). Two are available:

* ``mean``, the time average of each requested field. This is the default.
* ``rms``, the root mean square of the raw signal, :math:`\sqrt{\langle f^2 \rangle}`, computed **without subtracting the mean**. It is therefore the RMS of the signal itself and not the fluctuation intensity about the mean.

Both may be requested together: the mean is written under each field's normal name and the RMS under ``<field>_rms``. The two accumulate over independent windows. The RMS window opens when ``rms`` is first requested, so in a case forked from a parent that averaged without it, the mean continues the parent's window while the RMS covers only the time since the fork. Compare the two only when both windows span the same interval.

Statistics can be selected only on the volume, surface and slice time-averaged outputs (:py:class:`~flow360.TimeAverageVolumeOutput`, :py:class:`~flow360.TimeAverageSurfaceOutput` and :py:class:`~flow360.TimeAverageSliceOutput`). The other time-averaged outputs, namely isosurface, probe, surface probe, streamline and force distribution, write the mean only.

Requesting ``rms`` is subject to three restrictions, each enforced when the simulation parameters are validated:

* Imported surfaces average as mean only, so a time-averaged surface output that requests ``rms`` and includes an imported surface is rejected.
* All time-averaged volume outputs must request the same statistics as each other, and the same holds for time-averaged slice outputs, because the instances of each type merge into a single solver section. Surface outputs are exempt, since each one is translated separately.
* No requested output field may have a name ending in ``_rms``, which would collide with the name given to the RMS of the field of the same name.

.. _isosurfaceExtent_userGuide:

Limiting the Extent of an Isosurface
====================================

An isosurface is generated across the whole domain by default, which for quantities such as the Q criterion can produce a large surface where only one region is of interest. Two independent limits are available, and they can be combined:

* ``Clipping box`` (:py:attr:`~flow360.Isosurface.clipping_box`) keeps only the region of the isosurface inside a box, defined by its centre, size and rotation. It is set on each isosurface individually, so isosurfaces sharing an output can use different boxes, or none at all. The box is applied cell by cell: a cell is kept only if it lies entirely inside the box, so the clipped edge follows the mesh rather than the box faces exactly and can fall one cell short of them.
* ``wall_distance_clip_threshold`` (:py:attr:`~flow360.Isosurface.wall_distance_clip_threshold`) removes the part of the isosurface lying within a given distance of walls, which is useful for suppressing the near-wall sheet that vortex-identification quantities produce. Available only through the Python API.

Both are optional and neither is applied unless set. Limiting the extent reduces the size of the written surface as well as the visual clutter, since the discarded region is never written to file.

To configure the clipping box in each interface, see the :doc:`Isosurface Output </gui_guide/02.simulation-setup/04.output/02.outputs-list/11.isosurface-output>` page of the WebUI guide and the :doc:`output configuration </python_api/API_reference/outputs/output_index>` section of the Python API reference.

.. _sampleSurfaces_userGuide:

Sample Surfaces
===============

A sample surface (also called an imported surface) is a surface mesh that you supply yourself and use purely as an output location. It is a measuring device, not part of the simulation: it is never inserted into the volume mesh, it carries no boundary condition, and it has no effect whatsoever on the solution. Reach for one when you need flow data on an arbitrary non-planar surface, for example a curved plane crossing a duct or a streamtube-following cut, where one of the planar slice outputs listed under :ref:`availableOutputs_userGuide` will not do.

Importing a Sample Surface
--------------------------

In the WebUI, sample surfaces are imported from the **Resource** panel and then appear in the Entities browser. See the :doc:`Sample Surfaces </gui_guide/04.entities-browser/10.sample-surfaces>` page of the WebUI guide.

Through the Python API, import the file with :py:meth:`~flow360.Project.import_surface_mesh`, then register the returned handle on the draft through the ``imported_surfaces`` argument of :py:func:`~flow360.create_draft`. Only registered surfaces can be referenced when building the outputs. Already-imported surfaces are listed by :py:attr:`~flow360.Project.imported_surfaces`. See :ref:`python_api_import_sample_surface` for a complete working example.

The supported file formats are STL, CGNS and UGRID, and that list is exhaustive. A few properties of the file matter:

- A UGRID file must be accompanied by its ``.mapbc`` file. In the WebUI, upload both together.
- Volume mesh files in UGRID or CGNS format are also accepted, in which case their boundary faces are used as the sample surface.
- Files must be uncompressed. An archived file (for example ``.gz``) is rejected when the case runs.
- Every patch or zone in the file is merged into a single sample surface. Named boundaries inside a CGNS or UGRID file do not become separate output surfaces.
- The coordinates are used exactly as they appear in the file, in the coordinate frame and length unit of the volume mesh. No unit conversion is applied to them, so a file authored in different units has to be converted before it is imported.

A sample surface can be assigned to a coordinate system, which applies that system's translation, rotation and scaling to the surface. This is the supported way to reposition an imported file without editing it.

Where Sample Surfaces Can Be Used
---------------------------------

Sample surfaces are accepted by :class:`~flow360.SurfaceOutput`, :class:`~flow360.TimeAverageSurfaceOutput` and :class:`~flow360.SurfaceIntegralOutput`. They cannot be used anywhere else: not as a boundary condition, not for mesh refinement, and not in the remaining output types.

Two further restrictions apply:

- A single :class:`~flow360.SurfaceIntegralOutput` may contain either sample surfaces or regular simulation surfaces, but not both. Split them into separate outputs.
- Time averaging on a sample surface reports the mean only. The other statistics available to :class:`~flow360.TimeAverageSurfaceOutput` are rejected for sample surfaces.

How the Solution Is Sampled
---------------------------

The solution is not computed on a sample surface; it is interpolated onto it. Each node of the imported mesh is treated as a probe point, and its values are interpolated from the volume mesh cell that contains it. Three consequences follow:

- **The resolution is that of the volume mesh.** Refining the imported surface does not add detail. A finely tessellated sample surface sitting in a coarse region of the volume mesh resolves no more than the volume mesh does there.
- **Only volume fields are available.** The :ref:`surface-specific fields <outputFields_userGuide>` (such as ``Cf``, ``yPlus`` and ``heatFlux``) are wall quantities that require a boundary, so they are rejected on sample surfaces. The one surface quantity that is available is ``solution.node_unit_normal``, the outward unit normal of the imported mesh at each node.
- **A sample surface placed on a wall does not report wall data.** It reports the interpolated near-wall values from the volume mesh, which are not the same thing as the wall quantities produced by a regular surface output.

Sample Surfaces That Leave the Fluid Domain
-------------------------------------------

A sample surface is not checked against the geometry, and it is never trimmed to fit. It may pass straight through a solid body, extend beyond the farfield, overlap another sample surface, or be non-manifold. None of that is detected or reported when the case is submitted.

Instead, the outcome is decided node by node. A node that does not fall inside a fluid cell, because it is inside a solid body or outside the mesh altogether, is handled as follows:

- In a surface output, the node keeps its coordinates but all of its field values are written as ``NaN``. The affected region of the surface therefore appears as a hole when the file is opened.
- In a surface integral, the node is excluded from the sum. The reported integral covers only the part of the surface that lies inside the fluid domain, and nothing in the results file marks it as partial.
- The solver log records each skipped node with its coordinates, noting that the point is outside the grid volume and its output was skipped.

.. warning::
   Because a partial integral is reported as an ordinary number, a sample surface that unintentionally clips a wall or the farfield yields a quietly wrong result. Before trusting an integral, confirm from the solver log that no nodes were skipped, or open the matching surface output and check that it contains no ``NaN`` values.

Two behaviours are worth knowing when the surface interacts with moving parts of the domain:

- In a time-averaged output, a node that falls outside the fluid domain stays excluded for the whole averaging window.
- Nodes that land in a rotating zone are relocated at every physical step, so a stationary sample surface that cuts through a rotating zone continues to sample correctly as the mesh turns.

Integrating Over a Sample Surface
---------------------------------

A :class:`~flow360.SurfaceIntegralOutput` integrates a :class:`~flow360.UserVariable` over the sample surface. Define that variable as the **local** quantity at a point on the surface: the area weighting is applied for you, because each user variable assigned to a surface integral is multiplied by the local surface area before being summed. The integrated result is reported under the variable's name with an ``_integral`` suffix, in the corresponding integrated units.

A mass flow rate through a sample surface is therefore written as the local mass flux, the product of density and the velocity component along ``solution.node_unit_normal``, with no area factor of its own. Do not add one: ``solution.node_area_vector`` is a surface quantity and is rejected on a sample surface, so an expression using it fails validation.

Results are written to ``monitor_<output_name>_v2.csv`` and reached through ``case.results.monitors``, as for any surface integral.

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

.. _pressure_time_derivative:

Pressure Time Derivative
------------------------

``pressureTimeDerivative`` reports how fast the static pressure is changing at each point. Where pressure and its fluctuation show how strong the unsteady loading is, the time derivative shows how abruptly the pressure changes, which is what generates noise. Its root-mean-square over a time window therefore reads as a map of noise-source strength.

The field requires unsteady time stepping; requesting it in a steady simulation is rejected by validation. Add it to the ``Output fields`` list of Volume Output (:py:attr:`~flow360.VolumeOutput.output_fields`), Slice Output (:py:attr:`~flow360.SliceOutput.output_fields`), Volume Probe Output (:py:attr:`~flow360.ProbeOutput.output_fields`), Surface Output (:py:attr:`~flow360.SurfaceOutput.output_fields`), Surface Probe Output (:py:attr:`~flow360.SurfaceProbeOutput.output_fields`) or Surface Slice Output (:py:attr:`~flow360.SurfaceSliceOutput.output_fields`). It is not available as an isosurface field.

For the noise-source map, request it on a time-averaging output whose ``Statistics`` include ``rms`` (:py:attr:`~flow360.TimeAverageSurfaceOutput.statistics`, and the :py:attr:`volume <flow360.TimeAverageVolumeOutput.statistics>` and :py:attr:`slice <flow360.TimeAverageSliceOutput.statistics>` equivalents), which writes a separate ``pressureTimeDerivative_rms`` field. Being a scalar, it can also drive a stopping criterion through ``Monitor field`` (:py:attr:`~flow360.StoppingCriterion.monitor_field`).

Two behaviours are worth knowing. The derivative is computed in the solver using the same backward-difference formula that advances the solution in time, so one value is produced per physical time step and its accuracy follows ``Order of accuracy`` (:py:attr:`~flow360.Unsteady.order_of_accuracy`), except on the first physical step of a run, which falls back to first order. The field is also not exposed as a solver variable, so it cannot be used inside a :py:class:`~flow360.UserVariable` expression. To convert the values to Pa/s, see :ref:`Pressure Time Derivative <non_dim_pressure_time_derivative>`.

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
   - Hybrid models require unsteady simulations (they are not available for steady-state cases).

   These requirements are checked when the simulation is validated, so a
   mismatched request is reported against the offending output section
   before the case runs. The check covers every output section that can
   carry these fields, not just volume and slice outputs, and it reports
   which turbulence model and hybrid setting would be needed.

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

