"""
import_geometry_api.py - REST API wrapper for /v2/import-geometry endpoints
"""

import json
import os
import time

import requests

from flow360.cloud.rest_api import RestApi
from flow360.log import log

ENDPOINT = "v2/import-geometry"


class ImportGeometryApi:
    """API client for the import-geometry workflow."""

    def create(self, file_path, name=None):
        """
        POST /v2/import-geometry - Create an import-geometry resource.

        Returns dict with geometryId and uploadUrls.
        """
        if name is None:
            name = os.path.splitext(os.path.basename(file_path))[0]

        payload = {
            "files": [
                {
                    "name": os.path.basename(file_path),
                    "type": "main",
                }
            ],
            "name": name,
        }

        resp = RestApi(ENDPOINT).post(payload)
        return resp

    @staticmethod
    def _rewrite_presigned_url(presigned_url):
        """
        Rewrite the presigned URL to be reachable from the client.

        The backend may return presigned URLs with Docker-internal hostnames
        (e.g. minio:9000) that aren't reachable from the client. Replace the
        scheme+host+port with the s3_endpoint_url from the environment config,
        and return the original host so it can be sent as the Host header
        (required because the AWS signature includes the host).

        Returns (rewritten_url, original_host_or_None).
        """
        from urllib.parse import urlparse, urlunparse
        from flow360.environment import Env

        s3_endpoint = getattr(Env.current, "s3_endpoint_url", None)
        if not s3_endpoint:
            return presigned_url, None

        parsed_presigned = urlparse(presigned_url)
        parsed_endpoint = urlparse(s3_endpoint)

        # Only rewrite if the hosts differ
        if parsed_presigned.netloc == parsed_endpoint.netloc:
            return presigned_url, None

        original_host = parsed_presigned.netloc
        rewritten = urlunparse((
            parsed_endpoint.scheme,
            parsed_endpoint.netloc,
            parsed_presigned.path,
            parsed_presigned.params,
            parsed_presigned.query,
            parsed_presigned.fragment,
        ))
        log.debug(
            f"Rewrote presigned URL host: {original_host} -> {parsed_endpoint.netloc}"
        )
        return rewritten, original_host

    def upload_file(self, file_path, upload_url_info):
        """
        PUT file to presigned URL from upload_url_info['cloudpath'].
        """
        presigned_url = upload_url_info["cloudpath"]
        presigned_url, original_host = self._rewrite_presigned_url(presigned_url)

        headers = {}
        if original_host:
            # Preserve the original Host header so the presigned URL signature
            # validates (AWS signatures include the host).
            headers["Host"] = original_host

        with open(file_path, "rb") as f:
            response = requests.put(presigned_url, data=f, headers=headers)

        response.raise_for_status()
        log.debug(f"Uploaded {file_path} to presigned URL (status {response.status_code})")

    def complete_upload(self, geometry_id):
        """
        PATCH /v2/import-geometry/{id}/files - Mark upload as complete.
        """
        RestApi(ENDPOINT, id=geometry_id).patch(
            {"action": "Success"}, method="files"
        )

    def get_status(self, geometry_id):
        """
        GET /v2/import-geometry/{id} - Get current status.
        """
        return RestApi(ENDPOINT, id=geometry_id).get()

    def wait_until_processed(self, geometry_id, timeout=600, poll_interval=3):
        """
        Poll get_status until status is 'processed' or timeout is reached.
        """
        start = time.time()
        while time.time() - start < timeout:
            status_resp = self.get_status(geometry_id)
            status = status_resp.get("status", "")
            log.debug(f"Import geometry {geometry_id} status: {status}")

            if status == "processed":
                return status_resp

            if status == "error":
                raise RuntimeError(
                    f"Import geometry {geometry_id} failed with error: "
                    f"{status_resp.get('error', 'unknown')}"
                )

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Import geometry {geometry_id} did not finish within {timeout}s"
        )

    def fetch_tree(self, geometry_id):
        """
        GET /v2/import-geometry/{id}/tree - Fetch the hierarchical metadata tree.

        The response contains treeData as a JSON string that needs parsing.
        """
        resp = RestApi(ENDPOINT, id=geometry_id).get(method="tree")
        tree_data = resp.get("treeData", resp)
        if isinstance(tree_data, str):
            tree_data = json.loads(tree_data)
        return tree_data

    def save_face_grouping(self, geometry_id, config):
        """
        POST /v2/import-geometry/{id}/face-grouping - Save face grouping config.
        """
        return RestApi(ENDPOINT, id=geometry_id).post(config, method="face-grouping")

    def get_face_grouping_rules(self, geometry_id):
        """
        GET /v2/import-geometry/{id}/face-grouping-rules - Retrieve face grouping rules from S3.
        """
        return RestApi(ENDPOINT, id=geometry_id).get(method="face-grouping-rules")
