import flow360 as fl

# List all projects with "Simple Airplane" in their name
fl.Project.show_remote("Simple Airplane")

# Expected output:
# [14:14:18] INFO: >>> Projects sorted by creation time:
#             Name:         Simple Airplane from Python
#             Created at:   2024-10-17 10:59
#             Created with: Geometry
#             ID:           prj-fbc03e78-dd9e-4e0e-a527-dbd8195e120a
#             Link:         https://flow360.simulation.cloud/workbench/prj-fbc03e78-dd9e-4e0e-a527-dbd8195e120a
#             Geometry count:     1
#             Surface Mesh count: 1
#             Volume Mesh count:  1
#             Case count:         1
# ...
# [14:14:18] INFO: Total number of matching projects on the cloud: 6
