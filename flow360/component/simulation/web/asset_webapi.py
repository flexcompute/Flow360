"""
Thin asset web API wrappers.
"""

from __future__ import annotations

from flow360.component.interfaces import (
    CaseInterfaceV2,
    GeometryInterface,
    SurfaceMeshInterfaceV2,
    VolumeMeshInterfaceV2,
)
from flow360.component.simulation.web.resource_webapi import ResourceWebApi


class AssetWebApi(ResourceWebApi):
    """Thin wrapper around a single asset endpoint."""

    def __init__(self, interface, asset_id: str):
        self.asset_id = asset_id
        super().__init__(interface, asset_id)


class GeometryWebApi(AssetWebApi):
    """Thin geometry web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(GeometryInterface, asset_id)


class SurfaceMeshWebApi(AssetWebApi):
    """Thin surface mesh web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(SurfaceMeshInterfaceV2, asset_id)


class VolumeMeshWebApi(AssetWebApi):
    """Thin volume mesh web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(VolumeMeshInterfaceV2, asset_id)


class CaseWebApi(AssetWebApi):
    """Thin case web API wrapper."""

    def __init__(self, asset_id: str):
        super().__init__(CaseInterfaceV2, asset_id)
