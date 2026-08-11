.. _outputFields:
.. currentmodule:: flow360

Output Fields
-------------
This page presents all the output fields supported by different output types, including, the surface and volume solutions, as well as slices, isosurfaces, probe/surface-integral monitors and :class:`UserDefinedField`. 

Universal Fields
================
The following fields are supported by Surface, Volume, Slice and UserDefinedField types of output:

.. _UniversalVariablesV2:

.. csv-table::
   :file: ../tables/universalVariablesTable.csv
   :widths: 25,50
   :header-rows: 1
   :delim: @

Volume and Slice Specific Fields
======================================
The following fields are specifically supported by :class:`~flow360.VolumeOutput` and :class:`~flow360.SliceOutput`:

.. _VolumeAndSliceSpecificVariablesV2:

.. csv-table::
   :file: ../tables/volumeAndSliceSpecific.csv
   :widths: 25,50
   :header-rows: 1
   :delim: @

Surface Specific Fields
======================================
The following fields are specifically supported by :class:`~flow360.SurfaceOutput` and :class:`~flow360.SurfaceProbeOutput`:

.. _SurfaceSpecificVariablesV2:

.. csv-table::
   :file: ../tables/surfaceSpecific.csv
   :widths: 25,50
   :header-rows: 1
   :delim: @

IsoSurface Specific Fields
==========================
The following fields are specifically supported by :class:`~flow360.IsoSurfaceOutput`:

.. _IsoSurfaceSpecificVariablesV2:

.. csv-table::
   :file: ../tables/isosurfaceSpecific.csv
   :widths: 25,50
   :header-rows: 1
   :delim: @

User Variables
==============
Another way to define output fields is to use :class:`UserVariable` found in :ref:`User Defined Expressions <api_userDefinedExpressions>`.
User defined Variables can access multiple solver variables and undergo mathematical operations.
Please refer to :ref:`Variables <UserVariable>` section for more details about this feature as well as some usage examples.

User Defined Fields
======================================
Users can use the :class:`UserDefinedField` below to specify custom expressions to calculate outputs based on some solver variables. Please refer to the :doc:`this tutorial </python_api/example_library/notebooks/hinge_torques>` for more details.

.. autopydantic_model:: UserDefinedField
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource
