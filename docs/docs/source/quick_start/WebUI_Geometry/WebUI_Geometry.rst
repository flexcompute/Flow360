.. _quick_start_geometry:

Introduction to the Flow360 WebUI
***********************************

Welcome to Flow360. This tutorial introduces the Flow360 web-based user interface and the first step of a CFD simulation: uploading your CAD geometry.

.. _quick_start_geometry_download:

**Downloads**

Download the following CAD geometry file to your local machine:

`evtol_quickstart_grouped.csm <https://simcloud-public-1.s3.amazonaws.com/quickstart/evtol_quickstart_grouped.csm>`_

**Access Flow360**

Sign in to Flow360 at `flow360.simulation.cloud <https://flow360.simulation.cloud>`_.

If you do not have an account yet, contact our `sales team <sales@flexcompute.com>`_ to get an invitation code.

Uploading a geometry CAD file
==============================

Start by creating a new project. Click **New project** on the dashboard to open the **New project** window, then fill in the fields below. Each field is described in :doc:`/gui_guide/01.introduction/02.starting-project`.

* **Project name**: any name you like, for example ``eVTOL Quickstart``.
* **Solver version**: keep the preselected default release.
* **Project length unit**: the unit your CAD geometry was created in. The eVTOL geometry is in meters, so keep the default ``m``.
* **Description**: optional.
* **Creation methods**: keep **From geometry**.
* **Use CAD Importer V2**: leave it off. The beta mesher used in the :ref:`next tutorial <quickstart_meshing_webUI>` needs CAD Importer V1, which is also the importer that reads ``.csm`` files.

.. figure:: Figures/new_project_modal.png
   :width: 95%
   :align: center

   *The New project window.*

The formats Flow360 can import, with their file extensions, are listed in :doc:`/gui_guide/01.introduction/02.starting-project`.

Drop the **evtol_quickstart_grouped.csm** file :ref:`downloaded previously <quick_start_geometry_download>` into the drop area, or click **browse files** to select it. The file appears in the list below the drop area, ready to upload.

.. figure:: Figures/csm_dropped.png
   :width: 95%
   :align: center

   *The evtol_quickstart_grouped.csm file is ready to submit.*

Click **Submit**. The window closes, the file uploads in the background, and the new project appears under **Recent** on the dashboard. Once the upload and processing finish, click the project to open it in the workbench.

.. figure:: Figures/new_project_created.png
   :width: 95%
   :align: center

   *The new project on the dashboard.*

.. _fig_project_view:

.. figure:: Figures/geometry_uploaded.png
   :width: 95%
   :align: center

   *The uploaded eVTOL geometry.*

Because we uploaded a CAD geometry, the next step is to mesh it with the :ref:`automated meshing tool chain <quickstart_meshing_webUI>`. A project can also start from a surface mesh or a volume mesh instead. If you bring your own volume mesh, follow our :ref:`meshing guidelines <knowledge_base_meshing>`.

You can now proceed to the next quick start tutorial on :ref:`automated meshing <quickstart_meshing_webUI>`.
