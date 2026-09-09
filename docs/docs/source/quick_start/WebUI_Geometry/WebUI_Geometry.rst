.. _quick_start_geometry:

Introduction to the Flow360 WebUI
***********************************

Welcome to Flow360. The following provides a brief overview of the Flow360 web-based user interface and introduces the first step to a successful CFD simulation: Uploading your geometry CAD file. 

.. _quick_start_geometry_download:

**Downloads**

Please download the following CAD geometry file to your local machine:

`evtol_quickstart_grouped.csm <https://simcloud-public-1.s3.amazonaws.com/quickstart/evtol_quickstart_grouped.csm>`_

**Access Flow360**

Please sign in to Flow360 at `flow360.simulation.cloud <https://flow360.simulation.cloud>`_

If you do not have an account yet please contact our `sales team <sales@flexcompute.com>`_ to get an invitation code

Uploading a geometry CAD file
==============================

The first thing is to create a new project. To do that, please click on the **"New project"** icon to open up the **"New Project"** pop-up and fill in the desired name, and optionally a description. The **"Solver version"** will automatically be set to our latest version by default. The **"Unit"** is the dimensional unit that your CAD geometry was created in. In our case the default is m for meters.    

Alternatively, Flow360 also allows a user to directly upload a volume mesh as well. 

.. _fig_new_project:

.. figure:: Figures/new_project.png
   :width: 95%
   :align: center

   *New project icon and resulting pop up window.*

If you hover your cursor over the **Drag & drop files** area of the popup you will see a list of all the CAD formats we accept.


.. _fig_cad_formats:

.. figure:: Figures/cad_formats.png
   :width: 95%
   :align: center

   *Acceptable CAD formats.*

In this case we will upload a .csm script that defines our CAD geometry. Once the **evtol_quickstart.csm**  file :ref:`downloaded previously <quick_start_geometry_download>` has been dropped into the pop-up window you will notice that the file shows up at the bottom to confirm that it is ready to upload the CAD file to our servers.

.. _fig_csm_dropped:

.. figure:: Figures/csm_dropped.png
   :width: 95%
   :align: center

   *Confirmation that the evtol_quickstart.csm file is ready to submit.*

You are now ready to submit the CAD file to our servers by clicking on the **Submit** button. After a few seconds wherein the CAD file gets uploaded, and processed you will see that you have created a new project.

.. _fig_project_created:

.. figure:: Figures/new_project_created.png
   :width: 80%
   :align: center

   *New Project created.*

If you click on that window you will see your CAD ready to be analysed 

.. _fig_project_view:

.. figure:: Figures/geometry_uploaded.png
   :width: 95%
   :align: center

   *The uploaded EVTOL geometry.*

Since we are uploading a CAD geometry we will later invoke our :ref:`automated meshing tool chain <quickstart_meshing_webUI>`. You will notice that you can also upload your own volume mesh instead. If you choose to do so, please follow our :ref:`meshing guidelines <knowledge_base_meshing>`

You can now proceed to our next quickstart tutorial on :ref:`automated meshing <quickstart_meshing_webUI>`
