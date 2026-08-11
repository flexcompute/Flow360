.. _knowledge_base_stepCadFormat:

STEP Format CAD Import for Automated Meshing
**********************************************

Rather than building the geometry in ESP, the user can load a STEP file into the Engineering Sketch Pad (ESP), label the faces and edges in ESP, and save the labeled geometry as an EGADS file. Then, the EGADS file can be directly uploaded to our cluster via the Python API or using the Web UI.

Quick Example
==============

Here is an example CSM script for importing the `example.step <https://simcloud-public-1.s3.amazonaws.com/knowledge_base/step_cad_format/example.step>`_ file and labeling the faces.

.. code-block:: none
   :name: load_step_file_and_label_faces
   :emphasize-lines: 1-2, 16

   import $/example.step -1
   group -1
   store $prism
   store $cylinder
   store $box
   mark
      restore $prism
         attribute faceName $prism
         attribute groupName $prism
      restore $cylinder
         attribute faceName $cylinder
         attribute groupName $cylinder
      restore $box
         attribute faceName $box
         attribute groupName $box
   dump $/example.egads 0 1 0

#. For :code:`import` and :code:`dump`, the :code:`$/` means the STEP and EGADS files are in the same folder as the CSM file.
#. For :code:`import`, the :code:`-1` means import all solid bodies.
#. For :code:`group`, the :code:`-1` means ungroup the solid bodies so the user can :code:`store` them separately later on. The user can determine the order of the ungrouped solid bodies interactively in ESP.
#. For :code:`dump`, the :code:`0 1 0` means :code:`remove=false toMark=true withTess=false` i.e. all solid bodies from :code:`mark` will be exported into the EGADS file. More details about :code:`dump` can be found in the `ESP documentation <https://flexcompute.github.io/EngineeringSketchPad/EngSketchPad/ESP/ESP-help.html#dump>`_.

.. figure:: Figures/step_example_three_solid_bodies.svg
   :width: 80%

   Three solid bodies imported from a STEP file and labeled in ESP.

As shown in the example above, immediately after the user restores a solid body, the user can assign attributes to the solid body as well as all faces belonging to this solid body. For more options on selecting and attributing faces and edges, see :ref:`adding_face_and_group_attribute` and :ref:`adding_edge_attribute`

Currently face attributes from third-party CAD software will not be passed to our automated meshing workflow, though we are currently developing a feature to extract :code:`ADVANCED_FACE` names from STEP files, which will pass the face attributes through our workflow.

Common Issues
==============

This section summarizes common issues when using a STEP file as an entry point for automated meshing.

Lofting Surfaces
-----------------

When lofting the surfaces, it is strongly recommended to avoid sharp corners between edges or small gaps between adjacent surfaces.

.. figure:: Figures/sharp_corner_and_small_gap_on_surface.svg
   :width: 80%

   Example of a sharp corner and a small gap.

The schematic above shows two typical issues: (a) the sharp corner between surface-A/B/C could cause surface mesh failure on surface-B (b) the gap between surface-C/D can result in a surface mesh mismatch. It is worth noting that the size of the gap between surface-C/D is exaggerated for illustration. As shown in the screenshot below, in reality, the gap could be so small that it is almost invisible to the human eye.

.. figure:: Figures/invisible_gap.png
   :align: center
   :width: 90%

   Small/invisible gap on a sheet body in ESP.

.. figure:: Figures/mismatch_surface_mesh.png
   :align: center
   :width: 80%

   Surface mesh mismatch due to the small/invisible gap.

Boolean Operations
-------------------

When improving a design, changes often affect only a small portion of the geometry, leaving the rest unaltered. In such cases, users can import a baseline STEP file and then create a parametric geometry in ESP to represent the modified part. Before performing the :code:`UNION` operation, it is critical to ensure that the additional parametric geometry is fully extruded into the baseline geometry. This precaution prevents the occurrence of small gaps or sharp corners in 3D space.

.. figure:: Figures/3d_sharp_corner.svg
   :align: center
   :width: 100%

   Left\: Good example. The root cross-section of the pylon is completely inside the wing. Right\: Bad example, there is a sharp corner/small gap at the junction between pylon and wing, which could cause volume mesh failure.
   
Solid Body v.s. Sheet Body
---------------------------

Solid bodies output to STEP files are more robust than sheet bodies.

- In ESP, when dealing with sheet bodies, patching gaps/holes and "sewing" patches into a closed solid body is possible. However, this process heavily depends on the tolerance and may lead to issues during surface mesh generation. Hence, direct export of solid bodies to STEP files is strongly recommended.
- During surface mesh generation, we will always check the watertightness. Gaps or holes will trigger warning messages as follows: \"WARNING! There is only one neighboring face for an edge.\" As aforementioned, these gaps may be extremely small and invisible to the human eye.
-  As shown in the following screenshots, when preparing a half or quasi-3D model, it is recommended to keep the side faces on the "symmetric plane" and create solid bodies. The automated mesher will handle the side faces, eliminating the need for manual removal of these side faces by the user.

.. figure:: Figures/solid_vs_sheet_body.svg
   :align: center
   :width: 100%

   Left\: Good example. Solid body, side faces are kept. Right\: Bad example. Sheet body, the user does not need to create holes on the "symmetric plane"

Notes
------

Certain characters are not allowed anywhere in CSM scripts, including in comments:

- Vertical bar: :code:`|`

Using these characters may cause parsing errors or unexpected behavior.