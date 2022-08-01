# Standard Library
from itertools import combinations
from collections import namedtuple

# Third Party
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


InterpolatedPoints = namedtuple("InterpolatedPoints", "centroids simplices")


#########
# HELPERS
#########


def _build_simplicial_complex(points):
    triangulation = Delaunay(points)
    simplices = points[triangulation.simplices]

    return simplices # [num_simplices, 3, point_dim]


def _sort_simplices_by_area(simplices):
    AB = simplices[:, 1, :] - simplices[:, 0, :]
    AC = simplices[:, 2, :] - simplices[:, 0, :]
    areas = np.linalg.norm(np.cross(AB, AC)) / 2 # BUG: Cross-product returns wrong shape
    sort_indices = np.argsort(areas)

    return simplices[sort_indices]


def _filter_centroids_by_radius(centroids, points, radius):
    num_centroids = centroids.shape[0]
    valid_indices = []
    for i in range(num_centroids):
        centroid = np.expand_dims(centroids[i], axis=0)

        if not valid_indices:
            neighbors = points
        else:
            filtered_centroids = centroids[valid_indices]
            neighbors = np.concatenate((points, filtered_centroids), axis=0)

        dists = cdist(centroid, neighbors)
        if radius and dists.min() < radius:
            continue

        valid_indices.append(i)

    return valid_indices


def _interpolate_points(points, simplices, radius):
    simplicial_centroids = simplices.mean(axis=1)
    indices = _filter_centroids_by_radius(simplicial_centroids, points, radius)

    return InterpolatedPoints(
        centroids=simplicial_centroids[indices],
        simplices=simplices[indices]
    )


def _recursively_densify_points(points, radius, iter_results):
    """
    Helper function for triangular_densify.
    """

    simplices = _build_simplicial_complex(points)
    simplices = _sort_simplices_by_area(simplices)
    new_points = _interpolate_points(points, simplices, radius)
    iter_results.append(new_points)

    if not new_points:
        synthetic_points = np.concatenate([pts.centroids for pts in iter_results], axis=0)
        return synthetic_points, iter_results

    return _recursively_densify_points(
        np.concatenate((points, new_points.centroids), axis=0),
        radius,
        iter_results
    )


######
# MAIN
######


def densify(points, radius=1):
    """
    Parameters
    ----------
    points : numpy.ndarray
        an array of n-dimensional points representing the point cloud to
        upsample
    radius : int

    """

    return _recursively_densify_points(points, radius, iter_results=[])
