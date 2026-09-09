# Migration Guide

This folder contains useful tools to help migrating your old cases to use the latest Flow360 V2 API. 

## List of tools:

1. BET Disk converter - given a json case config file, extracts information about BET Disks and converts it to match the new version
2. Operating Condition from Mach number and muRef - creates an operating condition from Mach number and Reynolds number

## How to use BET Disk converter

Import `BETDisk` by using:
`from flow360.component.simulation.migration import BETDisk`

It contains the following functions:
- `read_single_v1_BETDisk()`
- `read_all_v1_BETDisks()`

`read_single_v1_BETDisk()` is used to convert a single a V1 (legacy) Flow360 input file into a single instance of `BETDisk` class suitable for use in the current version of Flow360.

`read_all_v1_BETDisks()` is used for extracting all BETDisks contained within a V1 (legacy) Flow360 input file into a list of `BETDisk` class instances suitable for use in the current version of Flow360.

Inputs can be in the form of either a single BETDisk instance json file using the following structure:
```
{
    "axisOfRotation": [
        ...
    ],
    "centerOfRotation": [
        ...
    ],
    ...
    "sectionalPolars": [
        ...
    ]
}
```
Or an entire flow360 json file structured like this:
```
{
    "geometry": {
        ...
    },
    ...
    "boundaries": {
        ...
    },
    "BETDisks": [
        ...
    ]
}
```


## How to use Operating Condition from Mach number and muRef

Import the `operating_condition_from_mach_muref` function by using:
`from flow360.component.simulation.migration.extra_operating_condition import operating_condition_from_mach_muref`

The function takes in the following inputs:
- mach
- muRef
- project_length_unit
- temperature
- alpha
- beta
- reference_mach

and returns an `AerospaceCondition` class instantiated based on given parameters.

Where `project_length_unit` is the length unit of the geometry or mesh used in the project.