"""
Heartbeat functions for usage during resource upload
"""

import time

from flow360.cloud.rest_api import RestApi

HEARTBEAT_INTERVAL = 15
TIMEOUT_MINUTES = 30


def post_upload_heartbeat(info):
    """
    Keep letting the server know that the uploading is still in progress.
    Server marks resource as failed if no heartbeat is received for 3 `heartbeatInterval`s.
    """
    while not info["stop"]:
        RestApi("v2/heartbeats/uploading").post(
            {
                "resourceId": info["resourceId"],
                "heartbeatInterval": HEARTBEAT_INTERVAL,
                "resourceType": info["resourceType"],
            }
        )
        time.sleep(HEARTBEAT_INTERVAL)
