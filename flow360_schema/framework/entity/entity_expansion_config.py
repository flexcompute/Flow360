"""Entity type expansion configuration for selectors."""

TargetClassExpansionMap = dict[str, list[str]]


DEFAULT_TARGET_CLASS_EXPANSION_MAP: TargetClassExpansionMap = {
    "Surface": [
        "Surface",
        "MirroredSurface",
        # The following types stay excluded for now to preserve current behavior.
        # They can be added later if the selector expansion surface grows.
        # "GhostSurface",
        # "WindTunnelGhostSurface",
        # "GhostSphere",
        # "GhostCircularPlane",
        # Note: ImportedSurface is excluded because it is only used for post-processing.
    ],
    "Edge": ["Edge"],
    "GenericVolume": ["GenericVolume"],
    "GeometryBodyGroup": [
        "GeometryBodyGroup",
        "MirroredGeometryBodyGroup",
    ],
}


MIRROR_CONTEXT_EXPANSION_MAP: TargetClassExpansionMap = {
    "Surface": [
        "Surface",
        "GhostSurface",
        "WindTunnelGhostSurface",
        "GhostSphere",
        "GhostCircularPlane",
        # MirroredSurface is excluded to prevent mirroring already-mirrored entities.
    ],
    "Edge": ["Edge"],
    "GenericVolume": ["GenericVolume"],
    "GeometryBodyGroup": [
        "GeometryBodyGroup",
        # MirroredGeometryBodyGroup is excluded.
    ],
}


def get_expansion_map(context: str = "default") -> TargetClassExpansionMap:
    """Return the expansion map for the requested selector expansion context."""
    if context == "mirror":
        return MIRROR_CONTEXT_EXPANSION_MAP
    return DEFAULT_TARGET_CLASS_EXPANSION_MAP


__all__ = [
    "DEFAULT_TARGET_CLASS_EXPANSION_MAP",
    "MIRROR_CONTEXT_EXPANSION_MAP",
    "TargetClassExpansionMap",
    "get_expansion_map",
]
