"""
Test script for GAI surface meshing via the Flow360 Python SDK, running
through the Catalyst pipeline end-to-end (geometry ingestion + GAI
surface meshing).

Creates a project from a .step geometry file, waits for ingestion, then
triggers a GAI surface mesh run and polls until completion.
"""

import sys
import time

import flow360 as fl
from flow360 import Env, MeshingParams, MeshingDefaults, SimulationParams, u

# ── Configuration ────────────────────────────────────────────────────────
GEOMETRY_FILE = "airplane.step"
LENGTH_UNIT = "m"
GEOMETRY_ACCURACY = 0.005  # in LENGTH_UNIT
SOLVER_VERSION = None  # None = use latest
WORKFLOW = "catalyst"  # "catalyst" or "standard"


# ── 1. Activate dev environment ──────────────────────────────────────────
Env.dev.active()
print("Environment: dev")
print(f"Workflow: {WORKFLOW}")

# ── 2. Create project from geometry (Catalyst or legacy workflow) ────────
print(f"\nCreating project from geometry: {GEOMETRY_FILE}")
project = fl.Project.from_geometry(
    GEOMETRY_FILE,
    name=f"gai-catalyst-test-{int(time.time())}",
    solver_version=SOLVER_VERSION,
    length_unit=LENGTH_UNIT,
    workflow=WORKFLOW,
)
print(f"  project_id:   {project.metadata.id}")
print(f"  project name: {project.metadata.name}")
print(f"  root item:    {project.metadata.root_item_type}")

# ── 3. Configure GAI meshing parameters ─────────────────────────────────
params = SimulationParams(
    meshing=MeshingParams(
        defaults=MeshingDefaults(
            geometry_accuracy=GEOMETRY_ACCURACY * u.m,
        ),
    ),
)
print(f"\nMeshing params: geometry_accuracy={GEOMETRY_ACCURACY} {LENGTH_UNIT}")

# ── 4. Generate surface mesh with GAI (inherits project workflow) ───────
print("\nSubmitting GAI surface mesh…")
surface_mesh = project.generate_surface_mesh(
    params=params,
    name="gai-catalyst-test-surface-mesh",
    run_async=True,
    use_geometry_AI=True,
)
print(f"  surface_mesh_id: {surface_mesh.id}")
print(f"  status:          {surface_mesh.status}")

# ── 5. Poll for completion ──────────────────────────────────────────────
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

# ── 6. Report ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("GAI Surface Mesh Result")
print(f"{'='*60}")
print(f"  project_id:       {project.metadata.id}")
print(f"  surface_mesh_id:  {surface_mesh.id}")
print(f"  final status:     {status_str}")
print(f"  workflow:         {WORKFLOW}")

if status_str == "processed":
    print("\nSUCCESS")
else:
    print(f"\nFAILED: {status_str}")
    sys.exit(1)
