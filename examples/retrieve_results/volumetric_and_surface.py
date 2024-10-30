import os
import tarfile
import tempfile

import flow360.component.v1 as fl
from flow360.examples import MonitorsAndSlices

MonitorsAndSlices.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(
    MonitorsAndSlices.mesh_filename, name="Volume-and-surface-mesh"
)
volume_mesh = volume_mesh.submit()

# submit case using json file
params = fl.Flow360Params(MonitorsAndSlices.case_json)
case = volume_mesh.create_case("Volume-and-surface-example", params)
case = case.submit()

# wait until the case finishes execution
case.wait()

results = case.results

with tempfile.TemporaryDirectory() as temp_dir:
    # download slice and volume output files as tar.gz archives
    results.slices.download(os.path.join(temp_dir, "slices.tar.gz"), overwrite=True)
    results.volumes.download(os.path.join(temp_dir, "volumes.tar.gz"), overwrite=True)

    # slices.tar.gz, volumes.tar.gz
    print(os.listdir(temp_dir))

    # extract slices file
    file = tarfile.open(os.path.join(temp_dir, "slices.tar.gz"))
    file.extractall(os.path.join(temp_dir, "slices"))
    file.close()

    # contains plots for all slices in the specified format (tecplot)
    # slice_x1.szplt, slice_y1.szplt
    print(os.listdir(os.path.join(temp_dir, "slices")))

    # extract volumes file
    file = tarfile.open(os.path.join(temp_dir, "volumes.tar.gz"))
    file.extractall(os.path.join(temp_dir, "volumes"))
    file.close()

    # contains volume plots in the specified format (tecplot)
    # volume.szplt
    print(os.listdir(os.path.join(temp_dir, "volumes")))
