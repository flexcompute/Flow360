.. _multielement_airfoil:
.. |br|     raw:: html

   <br/>

.. role:: orange

.. _DESPMTR: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#despmtr
.. _CFGPMTR: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#cfgpmtr
.. _CONPMTR: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#conpmtr
.. _MARK: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#mark
.. _SKBEG: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#skbeg
.. _SPLINE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#spline
.. _SKEND: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#skend
.. _GROUP: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#group
.. _STORE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#store
.. _DESPMTR: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#despmtr
.. _CONPMTR: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#conpmtr
.. _RESTORE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#restore
.. _JOIN: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#join
.. _ROTATEX: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#rotatex
.. _TRANSLATE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#translate
.. _COMBINE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#combine
.. _BLEND: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#blend
.. _EVALUATE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#evaluate
.. _PATBEG: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#patbeg
.. _IFTHEN: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#ifthen
.. _DIMENSION: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#dimension
.. _SELECT: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#select
.. _ESP help documentation: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#select
.. _SCALE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#scale
.. _EXTRUDE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#extrude
.. _ROTATEY: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#rotatey
.. _ATTRIBUTE: https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#attribute

.. _generateModel.py: https://simcloud-public-1.s3.amazonaws.com/multielement/generateModel.py
.. _multielement.model: https://simcloud-public-1.s3.amazonaws.com/multielement/multielement.model
.. _multielement_tutorials.tar.gz: https://simcloud-public-1.s3.amazonaws.com/multielement/multielement_tutorials.tar.gz
.. _multielementStep1.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep1.csm
.. _multielementStep2.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep2.csm
.. _multielementStep3.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep3.csm
.. _multielementStep4.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep4.csm
.. _multielementStep5.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep5.csm
.. _multielementStep6.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep6.csm
.. _multielementStep7.csm: https://simcloud-public-1.s3.amazonaws.com/multielement/multielementStep7.csm
.. _downloadCase.py: https://simcloud-public-1.s3.amazonaws.com/s809/downloadCase.py

.. currentmodule:: flow360

ESP Tutorial for a 2D Multi-element Airfoil
=============================================================================

In this tutorial, we will look at a 2D multi-element airfoil which consists of a slat, wing, and flap. We will demonstrate how to create a general multi-element configuration model in ESP and 
use the Flow360 Python API to run a RANS simulation. On this page you will find step-by-step instructions on how to:

#. Build a general quasi-3D multi-element configuration model in ESP
#. Set up automatic meshing and execute a RANS simulation
#. Use the Flow360 Python API to automatically update the configuration and re-submit a case

Firstly, we will explain how to build a general multi-element configuration model in Engineering Sketch Pad (ESP) and how to use edge, face, and group attributes to prepare that model for automatic meshing.
Then we will use that general model to build three standard test cases. The first case is a 3-element airfoil that is a cross-section of the NASA CRM-HL configuration and is used in a special session at AIAA Aviation 2020 called `Mesh Effects for CFD Solutions <https://ntrs.nasa.gov/citations/20205002818>`__.
There is no experimental data for this case and we only compare CFD results. The second case is 30P30N High-Lift Configuration, for which we have experimental data to compare to. The third case is the GA(W)-2 airfoil with a 30% fowler flap for which we also have experimental data.

Finally, we will set up the automatic meshing input files and use the Flow360 Python API to run a RANS simulation. The results of these simulations will then be validated against computational and experimental data.

.. figure:: Figures/30p30n/configuration.png
   :align: center
   :alt: quasi-3D model in ESP for a high-lift system configuration

   quasi-3D high-lift system configuration generated in ESP

.. note::

   In this tutorial, we demonstrate how to build a quasi-3D model for a general high-lift system configuration in ESP.
   The quasi-3D model here means extrusion of the 2D configuration.

Modeling Multi-element Configurations in ESP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESP uses the OpenCSM (Open-source Constructive Solid Modeller) system which is a feature-based, associative, parametric solid modeler.
More information about ESP is available `here <https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html>`__. To build the geometry through ESP, we use a series of statements and commands in a script. 
All ESP statements and commands are held in a \*.csm file.
In this tutorial, we follow a "bottom-up" approach to build a general model for a multi-element configuration including wing, flap and slat.
To build the model, we start by defining the design, configuration and constant parameters.

Defining Geometry Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The design parameter in this tutorial is the flap deflection angle. To define a design parameter, we use **DESPMTR**. 
Design parameters describe any particular instance of the geometry model. They can be single-valued, 1D vectors, or 2D arrays of numbers.
Each design value has a current value, upper- and lower-bounds. ESP allows to compute the sensitivity of any part of a geometry model with respect to
any design parameter.

.. rst-class:: esp-statement-block

   `DESPMTR`_    $pmtrName values
   use:    define a design Parameter

For the multi-element configuration, we also need to define the flap deflection axis location as a configuration parameter.
Since this is a quasi-3D model, the span of the quasi-3D model is also a configuration parameter.
To define a configuration parameter, we use **CFGPMTR**.
Values for the configuration parameters are declared in the top-level include-type .udc file. They must contain one or more numbers.
They must be dimension-ed, in case they are multi-valued. They have lower- and upper-bounds. Their values can be changed and sensitivities
cannot be computed for them.


.. rst-class:: esp-statement-block

   `CFGPMTR`_    $pmtrName value
   use:    define a configuration Parameter

To build a general model for the multi-element configuration, we need to consider that we may not have flaps, or slats or either.
Therefore, their presence in configuration parameters can be defined as a constant parameter for each. To define a constant parameter in a build process, we use **CONPMTR**.
Constant parameters contain only one number. Their values are declared in the top-level include-type .udc file and are visible from any .csm or .udc file.
Their values cannot be changed and sensitivities cannot be computed for constant parameters.

.. rst-class:: esp-statement-block

   `CONPMTR`_    $pmtrName values
   use:    define a constant Parameter

The example below shows how to put these statements all together in a CSM file.

In this example, **DIMENSION** indicates the size of the array used to define the flap deflection axis location.

.. rst-class:: esp-statement-block

    `DIMENSION`_ $pmtrName nrow ncol
    use:    set up or redimensions an array Parameter

.. code-block:: none
   :name: design_parameters
   :emphasize-lines: 2-3

   DESPMTR flapDeflection 0
   DIMENSION flapDefAxis 1 3 1
   CFGPMTR flapDefAxis "0.84;0.0;0.0109;"
   CFGPMTR span 0.01
   CONPMTR slatStatus 1
   CONPMTR flapStatus 1

The flap deflection angle is defined in degrees. The span is in meters. The status parameter for the flap and slat indicates if they are present in the configuration.
Any value other than zero indicates that they are present in the configuration.

Here you can find the complete CSM file for this step: `multielementStep1.csm`_.

3D Sketches of Multi-element Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we define a wire body for each element in the configuration. Then, we create a face (e.g. a higher type) from that wire body.

This section is similar to creating 3D sketches for airfoils in the :ref:`ONERA M6 Wing Geometry Modeling <OM6TutorialSketches>`.
To create a sketch body for each element, we use the following format:

.. code-block:: none
    :name: airfoil_3d_sketch
    :emphasize-lines: 2-8,10-16

    mark
    skbeg   up_x_TE  up_y_TE  up_z_TE
    spline  up_x_p1  up_y_p1  up_z_p1
    .
    .
    .
    spline  up_x_LE  up_y_LE  up_z_LE
    skend

    skbeg   low_x_LE  low_y_LE  low_z_LE
    spline  low_x_p1  low_y_p1  low_z_p1
    .
    .
    .
    spline  low_x_TE  low_y_TE  low_z_TE
    skend
    group
    store airfoil

The above CSM example shows how to put together a 3D sketch for an airfoil. These 3D sketches are later used to build a face body and then face bodies are used to be extruded into a solid body. 
In this example, we follow a simple concept to define a 2D curve. We define all the curves directly in global 3D space. In case, we need them on a plane, we define 3D data as needed.
For example, we define the multi-element configuration on the xz-plane. Therefore, we create all splines in the global 3D space by setting all y-coordinates of the 3D points to zero.

As shown in the above example, all 3D sketches start from the upper trailing-edge point to the leading-edge point and back to the lower trailing-edge point. This means the points definition is counter-clockwise.

In a multi-element configuration, we have three elements including the wing, which is the main element, the slat in front of the wing, and the flap in the back. The wing element profiles can have two curves defined through 
two splines, when there is a concave region in the back where the flap closes in. It can have more curves when the flap element is present. Since we define a generic multi-element modeler, we include all possible situations.
Therefore, wing element profiles always include two curves defined by two splines connected at the leading-edge. If the flap element is present in the configuration, 
we define a curve using a single spline in a separate sketch for the concave region in the back of the wing.

For the flap element, we have two curves defined by two splines in one sketch. For the slat element, we have two curves: the front and back curves. The front curve is defined with two spline in one sketch. 
The back curve is defined with one spline in a different sketch.
Please note that when we define a single spline in a sketch, we don't need to group it.

Here you can find the complete CSM file for this step including the sketches for three elements in the configuration: `multielementStep2.csm`_.

.. _wingFaceBody:

Building the Wing Face Body
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we **restore** the sketch body for each element and create a face from them.

.. rst-class:: esp-statement-block

    `RESTORE`_   $name index=0
    use:    restores Body(s) that was/were previously stored

In the above description, "=" indicates an optional argument for the statement. When the optional argument is not indicated, the default value after the equal sign is taken.

To create a face body from a wire body, we use **combine**.

.. rst-class:: esp-statement-block

    `COMBINE`_   toler=0
    use:    combine Bodys since Mark into next higher type

In this tutorial, we close the wire body and then use **join** to have one single wire body before calling the **combine** statement.

.. rst-class:: esp-statement-block

    `JOIN`_      toler=0 toMark=0
    use:    join two Bodys at a common Edge or Face

Because we create a general model for multi-element configuration, to close a wire body for each element, we have different scenarios.
Two possible scenarios to create a closed wire body for the wing element are explained below.

.. note::

   1. Wing Element Scenario A:

   The flap status is not equal to 1, which means there is no flap element in the configuration. Therefore, the wing wire body only includes the upper and the lower curves forming 
   an airfoil. To close the wire body, we can connect the upper and the lower trailing-edge points with a line sketch.

   .. figure:: Figures/examples/rae2822.png


   2. Wing Element Scenario B:

   The flap status is equal to 1, which means there is a flap element in the configuration. Therefore, the wing wire body includes three spline curves forming the
   upper wing, the lower wing and the concave curve in the rear portion of the wing. Based on the format explained in the :ref:`3D Sketch Format <airfoil_3d_sketch>`: 
   
      * The concave curve starts from the lower trailing-edge point and it ends up connected to the lower curve of the wing. In the figure below, as an example, the wing element for the 2D CRM high-lift configuration is shown. In this configuration, we have a flap element and the concave curve in the back of the wing is connected to the lower curve of the wing element, which creates a sharp point in the connection.

      .. figure:: Figures/crm/crmWire.png

      * The concave curve starts from the lower trailing-edge point and it does not end up connected to the wing lower curve. In the figure below, as an example, the wing element for the 30p30n high-lift configuration is shown. In this configuration, we have a flap element and as you see, the concave line in the back of the wing element is not connected to the lower curve of the wing. To create a closed wire body for the wing element, we need to connect the ending point of the concave curve to the ending point of the lower curve of the wing element.

      .. figure:: Figures/30p30n/30p30nWire.png

      In the figure below, the wing element for the GA(W)-2 high-lift configuration is shown. As you see, we have a flap element in this configuration and the concave curve in the back of the wing is not connected to 
      the lower curve of the wing. To create a close wire-body, we have to connect the ending point of the concave curve to the ending point of the lower curve of the wing. This is the second item in scenario B and it creates a blunt trailing-edge in the connection.

      .. figure:: Figures/gaw2/gaw2Wire.png

   To cover both types of curves in the concave rear part of the wing, first we connect the upper trailing-edge point to the 
   lower trailing-edge point and then in case the concave curve is not connected to the wing lower curve, we use a line sketch to connect it and close 
   the wire body for the wing.

Based on the above-mentioned scenarios, first we restore the wing sketch group and **join** it. Therefore, we have one wire body including
two edges and three nodes.

.. code-block:: none
   :name: wing_face_body1
   :emphasize-lines: 2

   restore wing
   join

Then, we must find the coordinates of these three nodes representing the leading-edge and the upper and lower trailing-edge. To find their coordinates, we loop over all the nodes in the
current body and detect the leading-edge node and the trailing-edge nodes based on their x and z values. The leading-edge node and the trailing-edge nodes for a wing wire with flap is shown below.

.. figure:: Figures/examples/wingEndings.png
   :align: center
   :alt: leading-edge and trailing-edge nodes for a wing profile

   Leading-edge and trailing-edge nodes for a wing profile


.. code-block:: tcl
   :name: wing_face_body2
   :emphasize-lines: 15-19, 26-29, 31-34

   # defining arrays with one row and three columns
   dimension wingLE 1 3
   dimension wingUpTE 1 3
   dimension wingLowTE 1 3

   # initializing xmin, zmin, zmax with their opposite extremes
   set xmin @xmax
   set zmin @zmax
   set zmax @zmin

   # looping over nodes to find the LE
   set nNodes @nnode
   patbeg inode nNodes
      evaluate node @ibody inode
      IFTHEN @edata[1] LT xmin
         set xmin @edata[1]
         set wingLE    "xmin;@edata[2];@edata[3];"
         set leadingNodeID inode
      ENDIF
   patend

   # looping over nodes to find the TE
   patbeg inode nNodes
      IFTHEN inode NE leadingNodeID
         evaluate node @ibody inode
         IFTHEN @edata[3] GT zmax
               set zmax @edata[3]
               set wingUpTE   "@edata[1];@edata[2];zmax;"
         ENDIF

         IFTHEN @edata[3] LT zmin
               set zmin @edata[3]
               set wingLowTE   "@edata[1];@edata[2];zmin;"
         ENDIF
      ENDIF
   patend

.. note::

    In this tutorial, a "bottom up" approach is provided. The solid body is built up from nodes, to curves, to edges, to surfaces, to faces, to shells, and finally bodies.
    The term 'node' here refers to the nodes at the ends of each edge, which differs from the airfoil data points used to define the upper and lower curves.
   
In the above code, first we define arrays for the leading-edge and trailing-edge nodes.
Then we assign the maximum value of the x-axis in this current body as the initial value for the :orange:`xmin` variable, 
the maximum value of the z-axis in this current body as the initial value for the :orange:`zmin` variable, and the minimum value of the
z-axis in this current body as the initial value for the :orange:`zmax` variable.

Then we collect the total number of nodes in this current body by using **node**, loop over them, **evaluate** them, and compare 
the x value of their coordinates with the :orange:`xmin` to find the leading-edge node. When x coordinate of a node has a value less than :orange:`xmin`,
the value of :orange:`xmin` is updated with that x value. We also store the nodeID at each update in the :orange:`leadingNodeID`. 
In this case, the leading-edge node is the most forward one, hence with the lowest x coordinate. Each wire body is composed of only edges and each edge has two nodes including the begin and end nodes.
When we collect all the nodes in the current body, we collect all the ending nodes for all of the edges. Please note the node here refers to the node as one of the topological entities and it is different
from the airfoil data points used to create the spline.

Then we loop over all nodes again and if the nodeID is not equal to :orange:`leadingNodeID`, we **evaluate** and compare the value of z coordinate to
find the upper trailing-edge node with maximum z and the lower trailing-edge node with minimum z. The way we find the leading-edge and trailing-edge nodes in this example is similar to adding edge attributes based on node coordinates in the :ref:`ONERA M6 Wing Geometry Modeling <OM6TutorialEdgeAttr>`.
For a detailed description you can refer to that tutorial.

After restoring the wing wire body, we restore the sketch spline for the concave line in the rear of the wing, if the flap status is not equal to zero.

Just like we did for the upper and lower curves above, we will now find the begin and end points of the concave spline.
Therefore, after we **restore** the concave spline sketch, we define two arrays for the coordinates of the starting and ending points, then we
initialize the :orange:`xmin` and :orange:`xmax` variables with the maximum and minimum x values of the current sketch body.
To find the starting point, which has minimum x value, and the ending point, which has the maximum x value, we loop over all the nodes and evaluate the coordinates
of each node. If the value for the x coordinate were less than the :orange:`xmin` variable, we update the :orange:`xmin` variable. We follow the same technique to find then ending point
with the :orange:`xmax` variable. The ending nodes of the concave spline is shown below in comparison to ending nodes of the wing profile.

.. figure:: Figures/examples/concaveEndings.png
   :align: center
   :alt: ending nodes of the concave spline

   Ending nodes of the concave spline for a wing profile.

.. code-block:: tcl
   :name: concave_curve
   :emphasize-lines: 1,16-19,21-24,28-30,33-37,39-45

   patbeg then ifzero(flapStatus,0,1)
      # restoring sketch
      restore wingConcave

      # defining arrays for the ending nodes
      dimension concaveEnd 1 3
      dimension concaveBegin 1 3

      # initializing xmin and xmax with their opposite extremes
      set xmax @xmin
      set xmin @xmax

      # looping over nodes to find the ending points
      patbeg inode @nnode
         evaluate node @ibody inode
         IFTHEN @edata[1] GT xmax
               set xmax @edata[1]
               set concaveEnd "xmax;@edata[2];@edata[3];"
         ENDIF

         IFTHEN @edata[1] LT xmin
               set xmin @edata[1]
               set concaveBegin "xmin;@edata[2];@edata[3];"
         ENDIF
      patend

      # connecting the wing upper TE node to the ending node of the wing concave curve with line segment
      skbeg concaveEnd[1] concaveEnd[2] concaveEnd[3]
         linseg wingUpTE[1] wingUpTE[2] wingUpTE[3]
      skend

      # calculating the distance between the wing lower TE node and starting node of the wing concave line
      set sum 0
      patbeg i 3
         set sum sum+(concaveBegin[i]-wingLowTE[i])^2
      patend
      set distance sqrt(sum)

      IFTHEN distance GT 1e-04
         # connecting the wing lower TE node to the starting node of the wing concave curve with line segment
         skbeg concaveBegin[1] concaveBegin[2] concaveBegin[3]
               linseg wingLowTE[1] wingLowTE[2] wingLowTE[3]
         skend
      ENDIF
   patend

After we have obtained the starting and ending point of the concave spline sketch, we connect the upper trailing-edge point of the wing to the ending node of the 
concave curve with a sketch and a line segment. Then we need to find out if the concave spline sketch is connected to the lower trailing-edge node of the wing spline sketch or not.
For that purpose, we calculate the distance between two nodes: the lower trailing-edge node of the wing spline sketch and the starting node of the concave spline sketch.
When this distance is zero, these two splines sketches are connected and share the same node; otherwise we define a sketch and use a line segment to connect them and close
the wire body.

Finally we use **combine** to elevate the wire body into a face, and then store it as a face.

.. code-block:: tcl
   :emphasize-lines: 1

   combine
   store wingFace

Here you can find the complete CSM file for this step: `multielementStep3.csm`_.

Building the Flap and Slat Face Bodies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After we built the wing face, we check the flap and slat status. If they are set to be present in the configuration, we build their faces.

.. code-block:: tcl
   :name: flap_curve
   :emphasize-lines: 1,19-23,30-33,35-38,43-45,50

   patbeg then ifzero(flapStatus,0,1)
      restore flap
      join

      # defining arrays for the leading-edge and trailing-edge nodes
      dimension flapLE 1 3
      dimension flapUpTE 1 3
      dimension flapLowTE 1 3

      # initializing xmin, zmin and zmax with their opposite extremes
      set xmin @xmax
      set zmin @zmax
      set zmax @zmin
      
      # looping over nodes to find the leading-edge node
      set nNodes @nnode
      patbeg inode nNodes
         evaluate node @ibody inode
         IFTHEN @edata[1] LT xmin
               set xmin @edata[1]
               set flapLE    "xmin;@edata[2];@edata[3];"
               set leadingNodeID inode
         ENDIF
      patend

      # looping over nodes to find the trailing-edge nodes
      patbeg inode nNodes
         IFTHEN inode NE leadingNodeID
               evaluate node @ibody inode
               IFTHEN @edata[3] GT zmax
                  set zmax @edata[3]
                  set flapUpTE   "@edata[1];@edata[2];zmax;"
               ENDIF

               IFTHEN @edata[3] LT zmin
                  set zmin @edata[3]
                  set flapLowTE   "@edata[1];@edata[2];zmin;"
               ENDIF
         ENDIF
      patend

      # closing the flap wire body by a line segment
      skbeg flapLowTE[1] flapLowTE[2] flapLowTE[3]
         linseg flapUpTE[1] flapUpTE[2] flapUpTE[3]
      skend

      # creating a face body from the wire body
      combine
      store flapFace
   patend

Like the :ref:`wing face <concave_curve>` example, after we **restore** the flap sketch group, we use **join** to have a single wire body.
Then, we use **dimension** to define arrays for the coordinates of the leading-edge and trailing-edge nodes. The we initialize :orange:`xmin`, :orange:`zmin` and :orange:`zmax` 
variables for the nodes in the flap wire body with the maximum value of x, maximum value of z and minimum value of z for the current body.
Then we loop over all the nodes, **evaluate** at each node, and compare the x value of the node to find the leading-edge node. We use the same technique 
to find the trailing-edge nodes. Then, we close the flap wire body with a line segment and finally elevate the wire body to the face and store it.

For the slat wire body, we follow the same approach. In comparison to the flap, the only difference is that the slat has two-spline sketches forming the front curve and one spline sketch in the back.

.. code-block:: tcl
   :name: slat_curve
   :emphasize-lines: 1,20-24,30-33,35-38,57-60,62-65,69-71,74-76,81

   patbeg then ifzero(slatStatus,0,1)
      # restore the front spline sketch
      restore slatFront
      join

      # defining arrays for the leading-edge and trailing-edge nodes
      dimension slatFrontLE 1 3 1
      dimension slatFrontUpTE 1 3 1
      dimension slatFrontLowTE 1 3 1

      # initializing xmin, zmin and zmax with their opposite extremes
      set xmin @xmax
      set zmin @zmax
      set zmax @zmin

      # looping over nodes to find the leading-edge node
      set nNodes @nnode
      patbeg inode nNodes
         evaluate node @ibody inode
         IFTHEN @edata[1] LT xmin
               set xmin @edata
               set slatFrontLE    "xmin;@edata[2];@edata[3];"
               set leadingNodeID inode
         ENDIF
      patend

      patbeg inode nNodes
         IFTHEN  inode NE  leadingNodeID
               evaluate node @ibody inode
               IFTHEN @edata[3] GT zmax
                  set zmax @edata[3]
                  set slatFrontUpTE   "@edata[1];@edata[2];zmax;"
               ENDIF

               IFTHEN @edata[3] LT zmin
                  set zmin @edata[3]
                  set slatFrontLowTE   "@edata[1];@edata[2];zmin;"
               ENDIF
         ENDIF
      patend

      # restore the back spline sketch
      restore slatBack

      # defining arrays for the trailing-edge nodes
      dimension slatBackUpTE 1 3 1
      dimension slatBackLowTE 1 3 1

      # initializing xmin, zmin and zmax with their opposite extremes
      set zmin @zmax
      set zmax @zmin

      # looping over nodes to find the trailing-edge nodes
      set nNodes @nnode
      patbeg inode nNodes
         evaluate node @ibody inode
         IFTHEN @edata[3] GT zmax
               set zmax @edata[3]
               set slatBackUpTE   "@edata[1];@edata[2];zmax;"
         ENDIF

         IFTHEN @edata[3] LT zmin
               set zmin @edata[3]
               set slatBackLowTE   "@edata[1];@edata[2];zmin;"
         ENDIF
      patend

      # closing the trailing edge at the bottom side of the slat
      skbeg slatFrontLowTE[1] slatFrontLowTE[2] slatFrontLowTE[3]
         linseg slatBackLowTE[1] slatBackLowTE[2] slatBackLowTE[3] 
      skend

      # closing the trailing edge at the top side of the slat
      skbeg slatFrontUpTE[1] slatFrontUpTE[2] slatFrontUpTE[3]
         linseg slatBackUpTE[1] slatBackUpTE[2] slatBackUpTE[3]
      skend

      # creating a face body from the wire body
      combine
      store slatFace
   patend

For the slat element, we have two edges defining the front curve and connected at the leading-edge, and one edge in the back.
In the above code, we use the same technique we used for the wing and flap elements to find the leading-edge and trailing-edge nodes for the slat element.
After that, we **restore** the slat's back spline sketch. It is a single spline; therefore we don't need to **join**.
Then, we define two arrays to find the upper and the lower nodes, initialize :orange:`zmin` and :orange:`zmax` variables, and loop over the nodes to find the upper and the lower nodes.
At the end, we use a line segment sketch to connect the trailing-edge nodes at the top of the slat and at the bottom side of the slat.
Now that the wire body is closed, we use **combine** to elevate the wire body into a face body.

Here you can find the complete CSM file for this step: `multielementStep4.csm`_.

Multi-element Configuration Solid Body
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we build the solid body for the each element from their face bodies.

.. code-block:: tcl
   :name: multielement_conf
   :emphasize-lines: 1-3,6,14

   # restoring faces to build solid body | wing
   restore wingFace
   extrude 0 span 0
   store wingBody

   # restoring faces to build solid body | flap
   patbeg then ifzero(flapStatus,0,1)
      restore flapFace
      extrude 0 span 0
      rotatey flapDeflection flapDefAxis[3] flapDefAxis[1]
      store flapBody
   patend

   # restoring faces to build solid body | slat
   patbeg then ifzero(slatStatus,0,1)
      restore slatFace
      extrude 0 span 0
      store slatBody
   patend

In the above code, for each element we **restore** their face and then we use the **extrude** statement to build the solid body.
After the extrusion, we store the body. The :orange:`span` parameter is defined in the preamble.

.. rst-class:: esp-statement-block

      `EXTRUDE`_   dx dy dz
      use:    create a Body by extruding an Xsect

Since we are building a generic model for a multi-element configuration and we have defined a flap deflection angle as a design parameter, after extruding the flap face body, we rotate the 
flap solid body around the flap deflection axis by using the **rotatey** statement.

.. rst-class:: esp-statement-block

      `ROTATEY`_   angDeg zaxis=0 xaxis=0
      use:    rotates Group on top of Stack around an axis that passes through (xaxis, 0, zaxis) and is parallel to the y-axis

Here you can find the complete CSM file for this step: `multielementStep5.csm`_.

Adding Edge Attribute
^^^^^^^^^^^^^^^^^^^^^

After building the solid models for the multi-element configuration, we must add edge attributes to each element and prepare them for automatic meshing.
To add the edge attribute, we use the node coordinates that were obtained above when we elevated the wire body into a face body.
To add edge attributes, we use EdgeAttr udprim. This udprim is shipped with the pre-built version of the ESP on Flexcompute's GitHub page.

We add edge attributes for each element after the extrusion and before the restore. For example for the wing element, we have two 
symmetric edges, one leading-edge and two trailing-edges.

.. code-block:: tcl
   :name: multielement_edgeAttr
   :emphasize-lines: 7,9,20-25,34,39

   # restoring faces to build solid body | wing
   restore wingFace
   extrude 0 span 0

   # adding edge attributes | wing
   dimension ySymm 2 1
   set ySymm "0;span;"

   patbeg i 2
      udparg edgeAttr attrname $edgeName  attrstr $symmetry
      udprim edgeAttr xyzs "wingLE[1];ySymm[i];wingLE[3];wingUpTE[1];ySymm[i];wingUpTE[3];"
      udparg edgeAttr attrname $edgeName  attrstr $symmetry
      udprim edgeAttr xyzs "wingLE[1];ySymm[i];wingLE[3];wingLowTE[1];ySymm[i];wingLowTE[3];"
      patbeg then ifzero(flapStatus,0,1)
         udparg edgeAttr attrname $edgeName  attrstr $symmetry
         udprim edgeAttr xyzs "concaveBegin[1];ySymm[i];concaveBegin[3];concaveEnd[1];ySymm[i];concaveEnd[3];"
         udparg edgeAttr attrname $edgeName  attrstr $symmetry
         udprim edgeAttr xyzs "wingUpTE[1];ySymm[i];wingUpTE[3];concaveEnd[1];ySymm[i];concaveEnd[3];"

         IFTHEN distance GT 1e-04
               udparg edgeAttr attrname $edgeName  attrstr $symmetry
               udprim edgeAttr xyzs "concaveBegin[1];ySymm[i];concaveBegin[3];wingLowTE[1];ySymm[i];wingLowTE[3];"
         ENDIF
      patend
   patend

   udparg edgeAttr attrname $edgeName  attrstr $wingleadingEdge
   udprim edgeAttr xyzs "wingLE[1];wingLE[2];wingLE[3];wingLE[1];span;wingLE[3];"
   udparg edgeAttr attrname $edgeName  attrstr $wingtrailingEdge
   udprim edgeAttr xyzs "wingUpTE[1];wingUpTE[2];wingUpTE[3];wingUpTE[1];span;wingUpTE[3];"
   udparg edgeAttr attrname $edgeName  attrstr $wingtrailingEdge
   udprim edgeAttr xyzs "wingLowTE[1];wingLowTE[2];wingLowTE[3];wingLowTE[1];span;wingLowTE[3];"

   patbeg then ifzero(flapStatus,0,1)
      udparg edgeAttr attrname $edgeName  attrstr $wingtrailingEdge
      udprim edgeAttr xyzs "concaveEnd[1];concaveEnd[2];concaveEnd[3];concaveEnd[1];span;concaveEnd[3];"
      udparg edgeAttr attrname $edgeName  attrstr $wingtrailingEdge
      udprim edgeAttr xyzs "concaveBegin[1];concaveBegin[2];concaveBegin[3];concaveBegin[1];span;concaveBegin[3];"
   patend

   store wingBody

For the wing element, we have three types of edges including: leading-edge, trailing-edge and symmetric.
We consider edges associated with the concave region of the multi-element configuration as trailing-edges when the flap status is nonzero.
Therefore, in the above code, we define an array with two elements including zero and span. These two elements define two symmetric planes for the 
multi-element configuration quasi-3D model. For the symmetric edges, we call the edgeAttr udprim twice. When the flap status is nonzero, 
we must also add the edge attribute for edges associated with the concave region. In the :ref:`Building the Wing Face Body <wingFaceBody>`, we 
defined and calculated a :orange:`distance` parameter to find out if the concave spline was connected to the wing lower trailing-edge node.
We use the same parameter here to add the edge attributes to associated edges in the concave rear part. This is for when the wing lower trailing-edge node is not 
connected to the starting node of the concave sketch spline.

For the leading-edge, we use the :orange:`wingLE` array defined in the above sections to add the edge attribute. For the trailing-edge, we use :orange:`wingUpTE` and :orange:`wingLowTE` arrays to 
add the edge attributes. When the flap status is nonzero, we use :orange:`concaveBegin` and :orange:`concaveEnd` arrays to add edge attributes to concave spanwise lines 
associated with the rear concave region of the wing element.

For the flap and slat elements, we follow the same procedure to add edge attributes to the leading-edge and trailing-edge.

Here you can find the complete CSM file for this step: `multielementStep6.csm`_.

Adding Face and Group Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After we added the edge attributes for each element, we must add the face and group attributes and prepare the model for the automatic meshing.

.. code-block:: tcl
   :name: multielement_faceAttr
   :emphasize-lines: 5-6,10,13,19,22-24

   # restoring bodies to add face and group attributes | wing
   restore wingBody

   # adding face and group attributes to the entire solid body
   attribute faceName $wing
   attribute groupName $wing

   set nWingFaces @nface
   set sumWingArea @area
   set minArea sumWingArea

   # finding the face with min are | wing trailing edge face
   patbeg i nWingFaces
      select face i
      IFTHEN @area LT minArea
         set minArea @area
         set trailingFaceID i
      ENDIF
   patend

   # adding face and group attributes to the wing trailing face
   select face trailingFaceID
   attribute faceName $wingTrailing
   attribute groupName $wing

In the above example, we **restore** the wing solid body and then we use **attribute** statement to add a string value 'wing' to the 
attribute name :orange:`faceName` and to add the same string value to the attribute name 'groupName'. The statement **attribute** is defined below.
When the first letter of the attribute value is a string, then it must be started with '$'.

.. rst-class:: esp-statement-block

      `ATTRIBUTE`_   $attrName attrValue
      use:    sets an Attribute for the Group on top of Stack

After adding the '$wing' as the attribute value to the attribute 'faceName' and 'groupName', we find the face with the minimum area and add '$wingTrailing' as the 
string value for the 'faceName' attribute. Like this, we can specify the maximum edge length for the surface mesh in the automatic meshing.
The 'groupName' attribute is same.

To find the face with the minimum area, we define a variable :orange:`minArea` and initialize it with the sum of all faces. 
Then, we loop over all faces and compare their face area. When their face area is less than the value of the :orange:`minArea`, we update the :orange:`minArea` variable and 
we store the faceID.

We follow the same procedure to add a string value to the 'faceName' and 'groupName' attributes.

Here you can find the complete CSM file for the multi-element configuration: `multielementStep7.csm`_.

Using Python to Add Coordinates and Build Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we have used ESP statements and written a script to build a general model for the multi-element configuration.
Now, we use a python script that lets us add the build parameters and data point coordinates for each element into the CSM script.

For this purpose, we divide the geometry modeling script with ESP statements into two parts: preamble and body. The preamble includes the build parameters 
and the sketch splines. The body includes the rest of the build process: **restore** sketch bodies, build the solid model, and add attributes 
to the solid body. Then, we save the body into a separate file and call it `multielement.model`_.

The `generateModel.py`_ takes a JSON input file (ie, input_configuration.json) that holds the build parameters and path to the data point coordinates for each element. 
Then, it adds them to the preamble of the `multielement.model` and saves the CSM file corresponding to the input JSON file. This CSM file later can be passed to ESP to build the model. 

To execute the build process using the python script, make sure the *generateModel.py* and *multielement.model* are in the same directory, then run:

.. code-block:: console

    $ python3 generateModel.py -json input_configuration.json -csm this_configuration.csm

The "input_configuration.json" holds input build parameters and the configuration's data coordinates. 
These data coordinates define the concave line in the rear part of the wing, the wing, flap, and slat.
Each element's data coordinates is defined in a separate file. For example, for the 2D CRM configuration, the input configuration JSON is:

.. raw:: html

   <details>
   <summary><a>Configuration JSON Example for 2D CRM System Configuration</a></summary>

.. code-block:: json

   {
      "coordinates": {
         "wing": "data/crmwing.dat",
         "concave": "data/crmconcave.dat",
         "flap": "data/crmflap.dat",
         "slat": "data/crmslat.dat"
      },
      "configuration": {
         "span": 0.01, 
         "flapDeflection": 0,
         "deflectionAxis": [0.76,0.0,-0.1],
         "slat": true,
         "flap": true
      }
   }

.. raw:: html

   </details>

|br|
The "this_configuration.csm" is the output of the *generateModel.py*. This file holds the build prescription for the configuration corresponding to the input configuration JSON file.
Users can later use it with ESP to build the complete model and check the build process.

When the :orange:`slat` and :orange:`flap` options are set to false in the input configuration JSON, only the quasi-3D model for the wing profile is created, which can be used to build a quasi-3D model for an airfoil.
All the associated files with this tutorial, including the input configuration JSON files, are available in `multielement_tutorials.tar.gz`_. By executing the *generateModel.py* with different input configuration JSON files, you can build different multi-element models in ESP.
Below we can see the three standard test cases that will be used for the simulations.

.. _defMeshJsonMultiElt:

.. figure:: Figures/30p30n/30p30nESP.png
   :align: center
   :alt: quasi-3D model in ESP for 30p30n system configuration

   Quasi-3D model generated in ESP 30p30n system configuration

.. figure:: Figures/crm/crmESP.png
   :align: center
   :alt: quasi-3D model in ESP for 2D CRM system configuration

   Quasi-3D model generated in ESP for 2D CRM system configuration

.. figure:: Figures/gaw2/gaw2ESP.png
   :align: center
   :alt: quasi-3D model in ESP for GA(W)-2 system configuration

   Quasi-3D model generated in ESP GA(W)-2 system configuration