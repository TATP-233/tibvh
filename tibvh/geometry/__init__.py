from . import aabb_generator
from . import geom_intersection
from . import mesh_intersection
from . import utils

# Import key functions for convenient access
from .aabb_generator import (
    compute_triangle_aabb,
    compute_geometry_aabb
)

from .geom_intersection import (
    ray_geom_intersection,
    ray_plane_intersection,
    ray_sphere_intersection,
    ray_box_intersection,
    ray_cylinder_intersection,
    ray_ellipsoid_intersection,
    ray_capsule_intersection
)

from .mesh_intersection import (
    ray_triangle_distance
)

from .utils import (
    _transform_ray_to_local,
    _transform_point_to_world
)

__all__ = [
    # Submodules
    "aabb_generator",
    "geom_intersection", 
    "mesh_intersection",
    "utils",
    
    # AABB generation functions
    "compute_triangle_aabb",
    "compute_geometry_aabb",
    
    # Ray-geometry intersection functions
    "ray_geom_intersection",
    "ray_plane_intersection", 
    "ray_sphere_intersection",
    "ray_box_intersection",
    "ray_cylinder_intersection",
    "ray_ellipsoid_intersection",
    "ray_capsule_intersection",
    
    # Mesh intersection functions
    "ray_triangle_distance",
    
    # Utility functions
    "_transform_ray_to_local",
    "_transform_point_to_world",
]

# Geometry type constants for convenience
GEOMETRY_TYPES = {
    'PLANE': 0,
    'SPHERE': 2, 
    'CAPSULE': 3,
    'ELLIPSOID': 4,
    'CYLINDER': 5,
    'BOX': 6,
    'MESH': 7
}

# Add geometry type constants to __all__
__all__.append("GEOMETRY_TYPES")