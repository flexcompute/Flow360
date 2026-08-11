.. meta::
    :description: user defined expressions knowledge base.

.. currentmodule:: flow360
.. _expressions_legacy_user_guide:

Legacy User Defined Expressions
===============================

.. note::
   This is an old feature. For now it is still usable but when defining outputs, please use User Variables instead of User Defined Fields. You can read more about User Variables and new User Defined Expressions here: :ref:`user_defined_expressions_user_guide`

User Defined Expressions are a powerful tool to interact with solver variables in order to expand the realm of what can be achieved. The major use of user defined expression falls into the following categories:

- :ref:`Simulate custom dynamics <user_defined_dynamic_user_guide>`.

- Specify initial condition in :class:`NavierStokesInitialCondition` and :class:`HeatEquationInitialCondition`.

- :ref:`Specifying flow condition on boundaries <surface_models>`. 


.. _UserDefinedFieldsGuidance:

Guidance on how to write User Defined Expressions
--------------------------------------------------

The expression follows syntax that is very similar to the C language. Although it should be mentioned that, for safety reasons, not all C language features are supported. Statements are separated by semicolon. Examples are available :doc:`here </python_api/example_library/notebooks/udd_alpha_controller>`  in our :ref:`tutorials <python_api_example_library>`.

The following operators and functions are allowed to define an expression:

Scalar function and operators
------------------------------
.. csv-table::
   :file:    Tables/scalarFunctionAndOperators.csv
   :widths: 20, 70
   :header-rows: 1
   :delim: @

.. caution::
   Please note that in all instances of User Defined Expressions integer divisions (i.e. :code:`"18/12"`) will discard the fraction part (which results in :code:`1` in this case) consistent with the C language. Please consider explicitly using at least one float number in division (i.e. :code:`"18.0/12"`) if such behavior is not desired.

Functions that operate on vectors 
--------------------------------------------------

Note that the vector being used in the following functions must have 3 components.

.. csv-table::
   :file:    Tables/vectorFunctionAndOperators.csv
   :widths: 40, 60
   :header-rows: 1
   :delim: @

.. _SupportedVariablesInUserExpression_:

Solver variables
-----------------

In an expression, users can use solution variables as well as custom variables. There are three categories of solver variables:

- Variables that are defined on all grid nodes and is not bound to a specific **"source"** (like zone or patch).
 
- Variables that are unique to the entire grid/simulation and are not bound to a specific **"source"** (like zone or patch). Unlike the first category, values of these variables are not related to particular grid nodes.

- Variables that are associated with a certain **"source"**. For example the :code:`CL` has to be associated with a surface patch; :code:`theta` has to be associated with certain volume zone(s).

.. note::
    All variables are non-dimensional. More information on nondimensionalization in Flow360 can be found at :ref:`non_dim_input_userGuide` and :ref:`non_dim_output_userGuide`.


Variables that are defined on all grid nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the variables that are defined on all grid nodes, the following solver variables can be used. These variables are listed as arrays since their values depends on the grid nodes. 

.. _ArraysNoSource_:

.. csv-table::
   :file:    Tables/variablesWithNOSourceArrays.csv
   :widths: 40, 30, 40
   :header-rows: 1
   :delim: |


Simulation wide variables
~~~~~~~~~~~~~~~~~~~~~~~~~~
For this second category, the following solver variables can be used. These variables are listed as scalars since they have a unique value across the entire grid/simulation. 

.. _ScalarsNoSource_:

.. csv-table::
   :file:    Tables/variablesWithNOSourceScalars.csv
   :widths: 40, 30, 40
   :header-rows: 1
   :delim: |

.. _patch_var_ude:

Zone or patch associated variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For variables that are associated with a certain zone or patch, the following solver variables can be used. For :code:`bet_thrust`, :code:`bet_torque` and :code:`bet_omega` user can add the index of the bet disk into the variable name such as :code:`bet_0_thrust` which indicates the thrust of the first bet disk. For actuator disk variables :code:`actuatorDisk_DISK_ENTITY_NAME_thrustMultiplier` and :code:`actuatorDisk_DISK_ENTITY_NAME_torqueMultiplier`, replace :code:`DISK_ENTITY_NAME` with the name of the actuator disk zone (e.g., :code:`actuatorDisk_propeller1_thrustMultiplier`).

.. note::
    The forces (:code:`forceX/Y/Z`) and moments (:code:`momentX/Y/Z`) should be distinguished from the force/moment coefficients (e.g. :code:`CL`, :code:`CMy`). The forces and moments are the numerators of the force/moment coefficients as shown in :ref:`tab_non_dim_coeff`.

All variables that are also listed in :ref:`universal variables<UniversalVariablesV2>` have the same definitions here. Some variables will not be available if the corresponding feature is not enabled. For example :code:`solutionTransition` is not available if the case does not use transition model.

.. _VarsWithSource_:

.. csv-table::
   :file:    Tables/variablesWithSource.csv
   :widths: 40, 20, 30, 40
   :header-rows: 1
   :delim: |


Explanation on the source type:
   - BET Disk ID: this is the index/position of the BET disk as they appear in the case JSON. Starting from 0.
   - Patch name: name of the patch defined in the mesh file.
   - Zone name: name of the volume zone defined in the mesh file.

Examples
--------

Examples of the usage of the legacy user defined expressions can be found in the :ref:`Python API Example Library <python_api_example_library>`:

- :doc:`F1 Car Demo <../../python_api/example_library/notebooks/F1_car_demo>` (:class:`~flow360.UserDefinedField`),
- :doc:`UDD Alpha Controller <../../python_api/example_library/notebooks/udd_alpha_controller>` (:class:`~flow360.UserDefinedDynamic`).
