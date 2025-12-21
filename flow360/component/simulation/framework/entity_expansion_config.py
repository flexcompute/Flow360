"""Entity type expansion configuration for selectors.

This module defines mappings from selector target classes to the actual entity type names
they should match when expanding selectors. This allows a single SurfaceSelector to match
multiple surface-related entity types (Surface, MirroredSurface, GhostSurface, etc.).
"""

from typing import Dict, List

# Type alias for expansion map
TargetClassExpansionMap = Dict[str, List[str]]

# Default expansion mapping for normal simulation context
DEFAULT_TARGET_CLASS_EXPANSION_MAP: TargetClassExpansionMap = {
    "Surface": [
        "Surface",
        "MirroredSurface",
        "GhostSurface",
        "WindTunnelGhostSurface",
        "GhostSphere",
        "GhostCircularPlane",
        # Note: ImportedSurface is excluded - it's only used for post-processing
    ],
    "Edge": ["Edge"],
    "GenericVolume": ["GenericVolume"],
    "GeometryBodyGroup": [
        "GeometryBodyGroup",
        "MirroredGeometryBodyGroup",
    ],
}

# Mirror context expansion mapping - excludes already-mirrored entity types
# to prevent circular references when performing mirror operations
MIRROR_CONTEXT_EXPANSION_MAP: TargetClassExpansionMap = {
    "Surface": [
        "Surface",
        "GhostSurface",
        "WindTunnelGhostSurface",
        "GhostSphere",
        "GhostCircularPlane",
        # Note: MirroredSurface is excluded to prevent mirroring already-mirrored entities
    ],
    "Edge": ["Edge"],
    "GenericVolume": ["GenericVolume"],
    "GeometryBodyGroup": [
        "GeometryBodyGroup",
        # Note: MirroredGeometryBodyGroup is excluded
    ],
}


def get_expansion_map(context: str = "default") -> TargetClassExpansionMap:
    """Get the appropriate expansion map for the specified context.

    Parameters:
        context: The context for expansion. Options:
                 - "default": Normal simulation context (DEFAULT_TARGET_CLASS_EXPANSION_MAP)
                 - "mirror": Mirror operation context (MIRROR_CONTEXT_EXPANSION_MAP)

    Returns:
        The expansion map for the specified context.
    """
    if context == "mirror":
        return MIRROR_CONTEXT_EXPANSION_MAP
    return DEFAULT_TARGET_CLASS_EXPANSION_MAP
