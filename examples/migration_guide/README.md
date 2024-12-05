# Migration Guide

This folder contains useful tools for helping you adjust your V1 cases to be V2 compatible.

## List of tools:

1. BET Disk converter - given a json case config file, extracts information about BET Disks and converts it to match the new version

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

They can later be used to, for example, upload them to webUI when creating BET Disk 3D model.