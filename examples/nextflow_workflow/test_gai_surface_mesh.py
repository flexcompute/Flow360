"""
Test script for GAI surface meshing workflow via the Flow360 Python SDK,
using the Catalyst pipeline for both geometry ingestion and surface meshing.

Uploads a .step geometry file, waits for ingestion, then triggers a GAI
surface mesh run and polls until completion.
"""

import sys
import time

from flow360 import Env, Geometry, MeshingParams, MeshingDefaults, SimulationParams, u
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import GeometryInterface
from flow360.component.project import Project

# ── Configuration ────────────────────────────────────────────────────────
GEOMETRY_FILE = "airplane.step"
LENGTH_UNIT = "m"
GEOMETRY_ACCURACY = 0.005  # in LENGTH_UNIT
SOLVER_VERSION = None  # None = use latest
WORKFLOW = "catalyst"  # "catalyst" or "standard"


def wait_for_geometry(geo_id: str, timeout_s: int = 600, poll_s: int = 10) -> str:
    """Poll until the geometry resource reaches a terminal state."""
    api = RestApi(GeometryInterface.endpoint)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        data = api.get(method=geo_id)
        status = data.get("status", "unknown")
        print(f"  geometry {geo_id[:12]}…  status={status}")
        if status == "processed":
            return status
        if status == "error":
            print(f"  ERROR detail: {data}")
            return status
        time.sleep(poll_s)
    print("  TIMEOUT waiting for geometry")
    return "timeout"


# ── 1. Activate dev environment ──────────────────────────────────────────
Env.dev.active()
print("Environment: dev")
print(f"Workflow: {WORKFLOW}")

# ── 2. Upload geometry ───────────────────────────────────────────────────
print(f"\nUploading geometry: {GEOMETRY_FILE}")
geo_draft = Geometry.from_file(
    file_names=[GEOMETRY_FILE],
    project_name=f"gai-catalyst-test-{int(time.time())}",
    length_unit=LENGTH_UNIT,
    solver_version=SOLVER_VERSION,
    workflow=WORKFLOW,
)
geo = geo_draft.submit(run_async=True)
geo_id = geo.id
project_id = geo.info.project_id
print(f"  geometry_id:  {geo_id}")
print(f"  project_id:   {project_id}")

# ── 3. Wait for geometry ingestion ───────────────────────────────────────
print("\nWaiting for geometry ingestion…")
status = wait_for_geometry(geo_id)
if status != "processed":
    print(f"\nGeometry ingestion failed: {status}")
    sys.exit(1)

# ── 4. Load project ─────────────────────────────────────────────────────
print(f"\nLoading project {project_id}…")
project = Project.from_cloud(project_id=project_id)
print(f"  project name: {project.metadata.name}")
print(f"  root item:    {project.metadata.root_item_type}")

# ── 5. Configure GAI meshing parameters ─────────────────────────────────
params = SimulationParams(
    meshing=MeshingParams(
        defaults=MeshingDefaults(
            geometry_accuracy=GEOMETRY_ACCURACY * u.m,
        ),
    ),
)
print(f"\nMeshing params: geometry_accuracy={GEOMETRY_ACCURACY} {LENGTH_UNIT}")

# ── 6. Generate surface mesh with GAI (Catalyst pipeline) ───────────────
print("\nSubmitting GAI surface mesh…")
surface_mesh = project.generate_surface_mesh(
    params=params,
    name="gai-catalyst-test-surface-mesh",
    run_async=True,
    use_geometry_AI=True,
)
print(f"  surface_mesh_id: {surface_mesh.id}")
print(f"  status:          {surface_mesh.status}")

# ── 7. Poll for completion ──────────────────────────────────────────────
#  .status auto-refreshes from the API when not in a terminal state.
print("\nPolling for surface mesh completion…")
deadline = time.monotonic() + 1800  # 30 min timeout
while time.monotonic() < deadline:
    sm_status = surface_mesh.status
    status_str = sm_status.value if hasattr(sm_status, "value") else str(sm_status)
    print(f"  status: {status_str}")
    if sm_status.is_final():
        break
    time.sleep(30)
else:
    print("  TIMEOUT waiting for surface mesh (30 min)")
    sys.exit(1)

# ── 8. Report ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("GAI Surface Mesh Result")
print(f"{'='*60}")
print(f"  geometry_id:      {geo_id}")
print(f"  project_id:       {project_id}")
print(f"  surface_mesh_id:  {surface_mesh.id}")
print(f"  final status:     {status_str}")
print(f"  workflow:         {WORKFLOW}")

if status_str == "processed":
    print("\nSUCCESS")
else:
    print(f"\nFAILED: {status_str}")
    sys.exit(1)
