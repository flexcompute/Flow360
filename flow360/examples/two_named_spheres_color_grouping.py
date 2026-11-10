"""
two-body, color-coded sphere STEP file example
"""

from .base_test_case import Asset, BaseTestCase, DownloadableAssets


class TwoNamedSpheresColorGrouping(BaseTestCase):
    """A single STEP file containing two named spheres, each split into green and
    red colored faces, used to demonstrate color-based face-grouping rules
    (``root.faces(color=...)``)."""

    name = "twoNamedSpheresColorGrouping"

    downloadable_assets = DownloadableAssets(
        geometry=Asset("local://two_spheres_cq_bodynamed.step"),
    )
