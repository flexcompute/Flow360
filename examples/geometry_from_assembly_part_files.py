import os

os.environ["FLOW360_BETA_FEATURES"] = "1"
import flow360 as fl

fl.Env.dev.active()

from flow360.component.geometry import Geometry

# 1. full assembly + parts files
full_assembly_parts = []
for filename in ["two_cubes.SLDASM", "cube_1.SLDPRT", "cube_2.SLDPRT"]:
    full_assembly_parts.append(os.path.join("./data/two_cubes", filename))
geometry_draft = Geometry.from_file(full_assembly_parts, name="testing-full-assembly-part", solver_version="geoHoops-24.7.0")
geometry = geometry_draft.submit()
print(geometry)

# 2. some part files are missing
full_assembly_parts = []
for filename in ["two_cubes.SLDASM", "cube_1.SLDPRT"]:
    full_assembly_parts.append(os.path.join("./data/two_cubes", filename))
geometry_draft = Geometry.from_file(full_assembly_parts, name="testing-missing-part", solver_version="geoHoops-24.7.0")
geometry = geometry_draft.submit()
print(geometry)

# 3. multiple root level files
full_assembly_parts = []
for filename in ["cube_1.SLDPRT", "cube_2.SLDPRT"]:
    full_assembly_parts.append(os.path.join("./data/two_cubes", filename))
geometry_draft = Geometry.from_file(full_assembly_parts, name="testing-root-level-files", solver_version="geoHoops-24.7.0")
geometry = geometry_draft.submit()
print(geometry)
