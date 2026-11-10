"""
two separate sphere STEP files example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TwoSpheresFaceGrouping(BaseTestCase):
    """Two separate single-sphere STEP files (each its own ModelFile), used to
    demonstrate file-scoped face grouping: one ``root.model_files(name=...)`` rule
    plus an ``is_remaining`` catch-all."""

    name = "twoSpheresFaceGrouping"

    downloadable_assets = DownloadableAssets(
        geometry=Asset(["local://sphere_r2mm.step", "local://sphere_r2mm_center_5_0_0.step"]),
    )
