import time
from flow360 import Env, Geometry
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import GeometryInterface

Env.dev.active()

geo = Geometry.from_file(
    file_names=["R6.stp"],
    project_name="nextflow-dev-test",
    solver_version="release-25.8.11",
    length_unit="mm",
    workflow="catalyst",
)
result = geo.submit(run_async=True)
geo_id = result.id
print(f"Submitted: {geo_id}")

# Poll for completion via REST API
api = RestApi(GeometryInterface.endpoint)
for i in range(60):
    data = api.get(method=geo_id)
    status = data.get("status", "unknown")
    upload_status = data.get("uploadStatus", "unknown")
    print(f"  [{i+1}/60] status={status}  uploadStatus={upload_status}")
    if status in ("processed", "error"):
        print(f"\nFinal status: {status}")
        break
    time.sleep(10)
else:
    print("\nTimeout waiting for geometry to finish")
