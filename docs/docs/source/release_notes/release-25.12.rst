release-25.12
==============

Released: 22 October 2026

A new version of Flow360, release-25.12, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features/Improvements*

#. New multi-species solver

   * Model gases other than air, mixtures with a fixed composition, or mixtures whose composition varies through the domain
   * Temperature-dependent specific heats from NASA 9-coefficient polynomials, with a built-in species library (N2, O2, Ar, He, H2, CO, CO2, CH4, H2O)
   * User defined dynamics can read species mass flows and drive inflow composition
   * Added support for specifying a gas other than air at ``Inflow`` boundary conditions

#. Added a compressibility correction to the SST turbulence model, which improves predictions of turbulent mixing at high Mach numbers
#. Added support for non-conformal fluid-solid interfaces in conjugate heat transfer (CHT) simulations
#. Improved robustness of the compressible isentropic solver at high CFL
#. Improved robustness for meshes with nodes shared by several zone interfaces, or by a zone interface and a periodic boundary
#. Added support for honoring the inner radius of the BET cylinder
#. Improved the error message when the input mesh has boundary faces with no boundary references

*Bug Fixes*

#. Fixed incorrect wall distance direction in rotating zones with the AFT transition model and turbulence models
#. Fixed incorrect unsteady results in nested rotation zones where an inner zone spins about a different axis from its parent zone

Post-processing
---------------

*New features*

#. Added the option to time-average vector outputs in the body frame or the global frame (``averaging_reference_frame``). For example, on a rotating disk the in-plane force components average to roughly zero in the global frame but stay non-zero in the body frame.

Geometry and Meshing
--------------------

*GeometryAI Surface Mesher*

#. Improved the deduplication of identical or nearly identical faces that caused artifacts in the output surface mesh
#. Made triangle sizes more uniform when ``target_surface_node_count`` is set to a low value
#. Reduced memory use by about 10%

*New Volume Mesher*

#. Reduced boundary layer meshing time (30% faster on a large test case)

*Bug Fixes*

#. Fixed bounding boxes from the V2 CAD importer that ignored the file's unit and the assembly's part placement. Re-import the geometry to apply the fix.
#. Fixed duplicated parts when an assembly is uploaded as multiple native CAD files with the V2 CAD importer
#. Fixed GeometryAI surface adaptation stopping early in some cases
#. Fixed unneeded fine refinement on outer surfaces caused by hidden geometry inside a body or across thin voids
#. Fixed CGNS volume mesh export failing on boundary or zone names that contain "/" or are "."
#. Fixed jobs stopping with no error message when an open edge lies on an unnamed patch. These jobs now fail with ERROR 4080, which lists the open edges.
#. Added a case log warning for zero- or negative-volume elements, and fixed the bad-node count in the case log for runs on multiple GPUs

Python Client
-------------

*Breaking changes*

#. Gas mixtures have a new layout, shared by thermally perfect gas and species solver simulations (impacts only users of this feature). ``Gas.species`` replaces ``Gas.species_transport_model``, and the ``FrozenSpecies`` and ``SpeciesTransportModel`` containers are removed. Species come from the built-in library with ``Species.from_library()`` or are defined with ``Species``. Whether the species are transported or kept at a fixed composition is now chosen on ``Fluid.species_transport_solver``, with ``FrozenSpeciesSolver`` or ``SpeciesTransportSolver``. A viscosity override moves to ``Gas.dynamic_viscosity``. Multi-species inputs written for 25.11 do not load and fail with an error pointing at the new layout. See :ref:`formulation_gas_mixtures`.
#. ``ThermallyPerfectGas`` is removed. A gas other than air is defined by its species, and a custom gas with constant specific heats by ``NASA9Coefficients.constant_gamma()``.
#. ``NavierStokesInitialCondition`` is renamed ``FluidInitialCondition`` and gains ``species_mass_fractions`` for the initial composition. The old name still loads as a deprecated alias.

*New features*

#. Added ``Case.stop()`` and ``flow360 case stop <case_id>`` to stop a running case. The case finishes as completed and keeps its results.
#. Added ``fl.PointGroup`` for importing many probe points at once, for example from a CSV file with ``PointGroup.from_csv()``. A group can be used in the probe outputs and as seeds of the streamline outputs.
#. The Python client now also trusts the certificates installed on the operating system, so it works behind a corporate proxy or with a private certificate authority without setting ``REQUESTS_CA_BUNDLE``. Set ``FLOW360_DISABLE_SYSTEM_CERTS=1`` to opt out.
#. Removed the upper version limits on the Python client's dependencies, so security fixes in those packages can be installed without waiting for a new Flow360 release
#. ``UserDefinedDynamic.output_target`` accepts a list of surfaces
#. Added validation that rejects output fields the solver cannot produce for the case, such as ``mutRatio`` without a turbulence model, ``betMetrics`` without a BET disk, ``wallFunctionMetric`` without a wall function boundary and ``residualHeatSolver`` without a ``Solid`` model
#. Added validation that rejects a ``RotationVolume`` whose ``enclosed_entities`` contains its own entity, which produced a wrong volume mesh. Saved projects with this setting are fixed when they are loaded.

*Bug Fixes*

#. Fixed user defined dynamics laws written with ``t`` failing to compile with ERROR 3030. ``t`` and ``physicalTime`` both name physical time and cannot be used as constant names.
#. Fixed log messages that contain ``%``, square brackets or emoji codes raising an error or being printed incorrectly
#. Fixed an explicit ``domain_type="full_body"`` being ignored when bodies are moved by a coordinate system, which left an unresolvable warning about the ``symmetric`` boundary
#. Fixed a box rotation written with the opposite sign of axis and angle being treated as a different box, which re-ran meshing that could have been reused

Web UI
------

*New features*

#. Mesh generation stop -- Added support for stopping surface and volume mesh generation.

   .. image:: Figures/stop-surface-mesh-generation.png
      :align: center
      :class: release-figure

#. Point entities from CSV -- Added support for importing CSV files to create point entities directly.

   .. image:: Figures/point-entities-from-csv.png
      :align: center
      :class: release-figure

#. Distance measurement -- Added a tool to measure the distance between two points.

   .. image:: Figures/measure-distance-between-two-points.png
      :align: center
      :class: release-figure

#. Colormap transparency -- Added a transparency setting for the visualization colormap.

   .. image:: Figures/colormap-transparency-setting.png
      :align: center
      :class: release-figure

#. Divergence points -- Added divergence points across the Geometry, Mesh, and Visualization tabs.

   .. image:: Figures/divergence-point-locations.png
      :align: center
      :class: release-figure

#. Aeroacoustic data download -- Added support for downloading all aeroacoustic postprocessing data.

   .. raw:: html

      <div class="release-video">
         <video controls preload="metadata" poster="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.12/download-aeroacoustic-data-cover.jpeg?width=1120&amp;name=download-aeroacoustic-data-cover.jpeg" src="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.12/download-aeroacoustic-data.mp4"></video>
      </div>

#. CHARM translator -- Added a translator from CHARM input files to Flow360 BET disks.

   .. image:: Figures/charm-bet-disk-translator.png
      :align: center
      :class: release-figure

#. Flow360 Report -- Introduced the new Flow360 Report: create customizable reports to compare simulation assets, synchronize views, and export results to PDF for easy sharing.

   .. raw:: html

      <div class="release-video">
         <video controls preload="metadata" poster="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.12/customizable-report-compare-and-export-cover.jpeg?width=1120&amp;name=customizable-report-compare-and-export-cover.jpeg" src="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.12/customizable-report-compare-and-export.mp4"></video>
      </div>

#. Unified viewpoint schema -- The viewpoint schema is now the same in the Web UI JSON and in Python code.

   .. raw:: html

      <div class="release-video">
         <video controls preload="metadata" poster="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.12/unified-viewpoint-schema-json-and-python-cover.jpeg?width=1120&amp;name=unified-viewpoint-schema-json-and-python-cover.jpeg" src="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.12/unified-viewpoint-schema-json-and-python.mp4"></video>
      </div>

#. Validation cases -- Added validation cases to the Example library.

   .. image:: Figures/validation-cases-in-example-library.png
      :align: center
      :class: release-figure

#. Geometry import and visualization -- The geometry viewer now shows CAD face colors for the V1 CAD importer, and ``_color`` is no longer available as a face grouping attribute. The geometry viewer no longer shows suppressed CAD components for the V2 CAD importer.
#. Performance -- Improved Web UI performance, with smoother interactions throughout the workflow. Fixed run-time force plots not loading correctly during or after a simulation. This covers surface forces, BET Disk, Actuator Disk, Porous Medium and ``ForceOutput``.

Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.12.*"

This will automatically select the latest patch version of 25.12.
