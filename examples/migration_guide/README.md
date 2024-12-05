# Migration Guide

This folder contains useful tools to help migrating your old cases to use the latest Flow360 V2 API. 

## List of tools:

1. BET Disk converter - given a json case config file, extracts information about BET Disks and converts it to match the new version
2. Operating Condition from Mach number and Reynolds number - creates an operating condition from Mach number and Reynolds number
3. Operating Condition from Mach number and muRef - creates an operating condition from Mach number and Reynolds number

## How to use BET Disk converter

Import the `bet_disk_convert` function from `bet_disk_converter` file, which is located in migration_guide folder.

The function takes in the following inputs:
- file
- save
- length_unit
- angle_unit
- omega_unit

and returns a tuple of lists:
- list of BET Disks
- list of Cylinder entities used for BET Disks

In order to save the BET Disk and Cylinder files as jsons, be sure to set `save=True`.

The files will appear in your current working directory.

They can later be used to create BET Disk 3D model on WebUI.

## How to use Operating Condition from Mach number and Reynolds number

You can use the function by typing `fl.operating_condition_from_mach_reynolds` and providing appropriate input parameters.

The function takes in the following inputs:
- mach
- reynolds
- project_length_unit
- temperature
- alpha
- beta
- reference_mach

and returns an `AerospaceCondition` class instantiated based on given parameters.

`project_length_unit` is the length unit of the geometry or mesh used in the project.

## How to use Operating Condition from Mach number and muRef

You can use the function by typing `fl.operating_condition_from_mach_muRef` and providing appropriate input parameters.

The function takes in the following inputs:
- mach
- muRef
- project_length_unit
- temperature
- alpha
- beta
- reference_mach

and returns an `AerospaceCondition` class instatiated based on given parameters.

`project_length_unit` is the length unit of the geometry or mesh used in the project.