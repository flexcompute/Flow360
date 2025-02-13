import flow360.v1 as fl

from flow360.log import log

INDENT = " " * 16

for surface_mesh in fl.MySurfaceMeshes(limit=10000):
    log.info(
        "\n" + "Surface mesh ID: " + str(surface_mesh.id) +
        "\n" + "Surface mesh name: " + str(surface_mesh.name) +
        "\n" + "Status: " + str(surface_mesh.status.value) +
        "\n" + "Volume meshes:"
    )
    for volume_mesh in fl.MyVolumeMeshes(surface_mesh_id=surface_mesh.id, limit=10000):
        print(f"{INDENT}Volume mesh ID: " + str(volume_mesh.id))
        print(f"{INDENT}Volume mesh name: " + str(volume_mesh.name))
        print(f"{INDENT}Status: " + str(volume_mesh.status.value))
