# Report Plugin

The **Report Plugin** provides automated PDF report generation for simulation results. It integrates with Flow360 (or similar frameworks) to pull data from simulation cases, produce tables and charts, and optionally render 3D images or screenshots. Users can easily configure how the PDF is structured, which items to include (e.g., summaries, inputs, tables, 2D and 3D charts), and how to style each item.

---

## Table of Contents
- [Report Plugin](#report-plugin)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Folder and File Structure](#folder-and-file-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Create or Load Simulation Cases](#1-create-or-load-simulation-cases)
    - [2. Prepare a Report Configuration](#2-prepare-a-report-configuration)
    - [3. Invoke the Plugin](#3-invoke-the-plugin)
  - [Configuration and Customization](#configuration-and-customization)
  - [Generating Screenshots](#generating-screenshots)
    - [Camera controls:](#camera-controls)
    - [Key Camera Attributes](#key-camera-attributes)
    - [Camera Examples](#camera-examples)
  - [Examples](#examples)
    - [Minimal Example](#minimal-example)

---

## Overview

This plugin:
- Loads simulation data from one or more **Case** objects.
- Extracts relevant metrics, such as velocities, forces, boundaries, and other fields.
- Creates custom sections (e.g. *Summary*, *Inputs*, *Tables*, *Charts*) and assembles them into a single LaTeX-based PDF document.
- Can optionally include 3D images or screenshots of the geometry, surfaces, or isosurfaces for an at-a-glance view of simulation geometry or results.

---

## Features

1. **Section-Based Reporting**  
   - Easily add sections like `Summary`, `Inputs`, and custom sections with specialized content.

2. **Tables**  
   - Collect data from one or multiple simulation cases and display in tabular form using the `Table` class.  
   - Built-in support for custom formatting (e.g., `.5g` or specialized formatting functions).

3. **2D Charts**  
   - Plot data such as forces, residuals, or general x-y plots with `Chart2D`.  
   - Supports logarithmic scaling if needed (e.g., for residual plots).  
   - Can combine multiple cases into one plot or generate separate plots per case.

4. **3D Visualization**  
   - Automatically generate (or embed) images of your 3D geometry (e.g., surfaces, isosurfaces, different camera angles).
   - Users can opt to generate screenshots of simulation geometry in various positions (requires an external service or local tool to produce the images; see [Generating Screenshots](#generating-screenshots) below).

5. **Flexible Layout**  
   - Choose how items are arranged (one item per page, multiple items per row, etc.).  
   - Control figure sizes, captions, new pages, etc.

---

## Folder and File Structure

A high-level view of the relevant files:

flow360/plugins/report/
├── report.py               # Main file that defines the ReportTemplate class 
├── report_items.py         # Contains classes for different report "items" like Summary, Table, Chart2D, Chart3D, etc.
├── report_context.py       # (Imported by others) Manages the context in which the report is generated
├── utils.py                # Utility functions for data extraction, data formatting, etc.
└── uvf_shutter.py          # (Optional) Helper for generating 3D geometry screenshots

Example files in the repo:

examples/
└── automotive_example_in_cloud.py   # Shows how to configure and instantiate a report

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install flow360 package:
   ```bash
   pip install flow360
   ```




---

## Usage

### 1. Create or Load Simulation Cases

You need one or more **Case** objects that contain all relevant simulation data. The plugin references fields in these cases to build tables and charts.

### 2. Prepare a Report Configuration

Typically, you have a JSON or Python-based config specifying which `ReportItem` objects you want in your report, e.g.:

```python
report = ReportTemplate(
  title="Aerodynamic analysis of DrivAer",
  items=[
    Summary(text="Short summary..."),
    Inputs(),
    Table(
      data=[
        "params/reference_geometry/area",
        DataItem(data="surface_forces/totalCD", operations=Average(fraction=0.1)),
        "params/time_stepping/max_steps",
      ],
      section_title="Statistical data",
    ),
    Chart2D(
        x="surface_forces/pseudo_step",
        y="surface_forces/totalCD",
        section_title="Drag Coefficient",
        fig_name="cd_fig",
    )
  ]
)
report.to_file('path/to/reportTemplate.json"')
```
(This might be stored in a file like reportTemplate.json.)

```JSON
{
    "include_case_by_case": false,
    "items": [
        {
            "text": "Short summary...",
            "type_name": "Summary"
        },
        {
            "type_name": "Inputs"
        },
        {
            "data": [
                "params/reference_geometry/area",
                {
                    "data": "surface_forces/totalCD",
                    "operations": [
                        {
                            "fraction": 0.1,
                            "type_name": "Average"
                        }
                    ],
                    "type_name": "DataItem"
                },
                "params/time_stepping/max_steps"
            ],
            "section_title": "Statistical data",
            "type_name": "Table"
        },
        {
            "fig_name": "cd_fig",
            "section_title": "Drag Coefficient",
            "type_name": "Chart2D",
            "x": "surface_forces/pseudo_step",
            "y": "surface_forces/totalCD"
        }
    ],
    "title": "Aerodynamic analysis of DrivAer"
}
```



### 3. Invoke the Plugin
Use the main ReportTemplate class (from report.py) to build and export your PDF. For example (similar to automotive_example_in_cloud.py):

```python
from flow360.plugins.report.report import ReportTemplate
from flow360.component.case import Case

# Suppose you have a list of Case objects
cases = [...]

# Point to your config:
report = ReportTemplate(filename="path/to/reportTemplate.json")

# Generate the PDF:
report.create_pdf(
    output_folder="my_report_output",
    cases=cases
)
```


## Configuration and Customization

- **Report Items**: The core building blocks, each corresponding to a section or figure in the PDF.
  - *Summary*: Adds a textual overview.  
  - *Inputs*: Collects key input parameters from your cases (e.g., velocity, time-stepping).  
  - *Table*: Aggregates data from multiple cases into one table.  
  - *Chart2D*: Produces an XY plot from data paths in your case(s).  
  - *Chart3D*: Optionally display geometry or color-field surfaces (like `Cp`, `yPlus`, etc.).

- **Custom Data Paths**: Many items accept `data` arrays where each entry references a JSON path inside each case, e.g. `"params/operating_condition/velocity_magnitude"`.  
  - If the plugin finds that path in each case’s data, it extracts the values and populates the table/plot.
  
- **Formatting**:  
  - Use built-in format specifiers or pass a custom function to format numeric data.  
  - For tables, set `formatter` as a string (e.g. `".5g"`) or a callable for specialized formatting.

- **Multiple Cases**: If you provide multiple cases in a single run, the plugin can:
  - Combine them side by side in tables.  
  - Overlay them on the same chart.  
  - Generate separate pages or subplots (use `separate_plots=True` or `items_in_row=...` to configure layout).



## Generating Screenshots

The plugin can produce screenshots of your simulation geometry or results. You can activate this in `Chart3D` items by specifying a field or object type to display. The actual screenshot generation leverages an external tool, but the plugin manages how the images are embedded in the final PDF.


### Camera controls:

The `Camera` class specifies a 3D viewpoint and zoom for rendering geometry or visualizing simulation results. By adjusting properties such as `position`, `look_at`, and `dimension`, users can control the angle, orientation, and scale of the view. Below are the attributes you can configure (note that all lengths are in the same units used in your geometry or volume mesh):


### Key Camera Attributes

- **Position**  
  The camera’s eye position in 3D space. Typically, you can think of it as a point on a sphere, looking inward toward `look_at`. Adjusting `position` can rotate the view around the object of interest.

- **Up**  
  A vector that determines which way is “up” in the final image. Changing this can rotate the view around the axis from the camera’s position to the `look_at` point.

- **Look At**  
  The point at which the camera is aimed. By default, if unspecified, it’s often the bounding-box center of your model.

- **Pan Target**  
  A specific point to which you want to pan your camera’s center. If left unset, `look_at` remains the default center of view.

- **Dimension & Dimension Direction**  
  Together, these control the zoom level. For example, if `dimension_dir` is `"width"` and `dimension` is `2.0`, the rendered view’s width represents 2 model units. Similarly, use `"height"` or `"diagonal"` to scale in those directions.


---

### Camera Examples

<img width="500" src="https://github.com/user-attachments/assets/1a0df86e-ff1d-423f-958e-efc0b250783b" />

See how changing `position` and `look_at` affects the orientation of the camera.

<img width="500" src="https://github.com/user-attachments/assets/1cd4fbec-fa30-40c8-8946-73ca01d32bf0" />

See how `pan_target` changes the view.

<img width="500" src="https://github.com/user-attachments/assets/174043ae-f0f8-4100-bba2-af75a4b32755" />

See the effect of `dimension` and `dimension_dir` on zoom level.




## Examples

### Minimal Example

```python
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import *

cases = [...]  # Load or define your cases

report = ReportTemplate(
  title="Aerodynamic analysis of DrivAer",
  items=[
    Summary(text="Short summary..."),
    Inputs(),
    Table(
      data=[
        "params/reference_geometry/area",
        DataItem(data="surface_forces/totalCD", operations=Average(fraction=0.1)),
        "params/time_stepping/max_steps",
      ],
      section_title="Statistical data",
    ),
    Chart2D(
        x="surface_forces/pseudo_step",
        y="surface_forces/totalCD",
        section_title="Drag Coefficient",
        fig_name="cd_fig",
    )
  ]
)

# Save the JSON config or pass it directly:
report.create_pdf(
  "my_report",
  cases
)
report.wait()
report.download("my_report.pdf")
```


After running, check the my_report.pdf for your compiled PDF.

More Advanced
See `automotive_example_in_cloud.py` for more advanced usage (multiple 3D fields, custom cameras, integration with a pipeline, etc.).


---

**Enjoy creating automated simulation reports with the Report Plugin!**
