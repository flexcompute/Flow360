"""
Gzip compression of large webapi request bodies.
"""

import gzip
import json
from typing import Optional

from ..log import log

BYTES_PER_MB = 1024 * 1024

# zlib's default level, and what the web UI's CompressionStream("gzip") uses. Level 9
# spends several times the CPU on a multi-hundred-MB simulation.json for a ratio gain of
# about one percent.
GZIP_COMPRESS_LEVEL = 6

GZIP_HEADERS = {"Content-Type": "application/json", "Content-Encoding": "gzip"}


def compress_json_body(body, threshold_mb: Optional[float]) -> Optional[bytes]:
    """
    Gzip a JSON request body once it grows past ``threshold_mb``.

    Returns None when compression is off (``threshold_mb`` is None) or the body is at or
    below the threshold, so the caller keeps handing the body to ``requests`` as ``json=``
    and nothing about the request changes.

    ``allow_nan=False`` mirrors what ``requests`` does for ``json=``, so a body holding
    NaN or Infinity is rejected here exactly as it would be on the uncompressed path
    instead of being gzipped into JSON no parser accepts.
    """
    if threshold_mb is None:
        return None

    encoded = json.dumps(body, allow_nan=False).encode("utf-8")
    if len(encoded) <= threshold_mb * BYTES_PER_MB:
        return None

    compressed = gzip.compress(encoded, compresslevel=GZIP_COMPRESS_LEVEL)
    log.debug(f"gzip request body: {len(encoded)} bytes -> {len(compressed)} bytes")
    return compressed
