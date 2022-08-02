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
    """
    Takes a set of points in n-dimensional space and uses Delaunay triangulation
    to build a simplicial complex that covers the convex hull of the set.

    Parameters
    ----------
    points : numpy.ndarray
        set of points of shape [num_points, n]

    Returns
    -------
    simplices : numpy.ndarray
        array of simplices representing the complex, defined by their vertices;
        shape of [num_simplices, num_vertices, n]
    """

    triangulation = Delaunay(points)
    simplices = points[triangulation.simplices]

    return simplices


def _calculate_volume_of_simplex(simplex):
    """
    Given a set of vertices, this calculates the volume of the simplex that is
    the convex hull of those vertices. Computed using the Gram-determinant.

    Parameters
    ----------
    simplex : numpy.ndarray
        array of n-dimensional vertices representing a simplex; shape of
        [num_vertices, n]

    Returns
    -------
    volume : float
        volume of the simplex
    """

    gramian = np.linalg.det(np.dot(simplex, simplex.T))
    coef = 1 / np.math.factorial(simplex.shape[0])
    volume = coef * np.sqrt(gramian)

    return volume


def _sort_simplices_by_volume(simplices):
    """
    Sorts an array of simplices in descending order of volume. This is used to
    to maximize the amount of oversampling permitted by the radius constraint.

    Parameters
    ----------
    simplices : numpy.ndarray
        array of simplices defined by their vertices

    Returns
    -------
    sorted_simplices : numpy.ndarray
        same array of simplices but sorted in order of descending volume
    """

    num_simplices = simplices.shape[0]
    volumes = np.zeros(num_simplices)
    for i in range(num_simplices): # TODO: Use vector ops instead of a loop
        simplex = simplices[i]
        volumes[i] = _calculate_volume_of_simplex(simplex)

    sort_indices = np.argsort(volumes)
    sorted_simplices = simplices[sort_indices[::-1]]

    return sorted_simplices


def _filter_centroids_by_radius(centroids, points, radius):
    """
    Finds simplex centroids (i.e. candidates for interpolation) which are
    outside a given radius of all other points in the cloud.

    Parameters
    ----------
    centroids : numpy.ndarray
        array of simplex centroids
    points : numpy.ndarray
        array of points representing the current state of the point cloud
    radius : float
        minimum distance between interpolated points and existing points

    Returns
    -------
    filtered_indices : numpy.ndarray
        array of indices of centroids which satisfy the radius constraint
    """

    num_centroids = centroids.shape[0]
    filtered_indices = []
    for i in range(num_centroids):
        centroid = np.expand_dims(centroids[i], axis=0)

        if not filtered_indices:
            neighbors = points
        else:
            filtered_centroids = centroids[filtered_indices]
            neighbors = np.concatenate((points, filtered_centroids), axis=0)

        # TODO: Filter the list of neighbors to compare with
        dists = cdist(centroid, neighbors)
        if radius and dists.min() < radius:
            continue

        filtered_indices.append(i)

    return np.array(filtered_indices)


def _interpolate_points(points, simplices, radius):
    """
    "Densifies" a set of points by calculating the centroids of the simplices
    defined by those points and filtering out the centroids that violate the
    radius constraint.

    Parameters
    ----------
    points : numpy.ndarray
        array of points representing the current state of the point cloud
    simplices : numpy.ndarray
        array of simplices defined by their vertices; shape of
        [num_simplices, num_vertices, n]
    radius : float
        minimum distance between interpolated points and existing points

    Returns
    -------
    interpolated_points : tuple
        tuple of centroids and the simplicial complex the centroids are derived
        from
    """

    simplicial_centroids = simplices.mean(axis=1)
    indices = _filter_centroids_by_radius(simplicial_centroids, points, radius)

    if not indices.any():
        return None

    interpolated_points = InterpolatedPoints(
        centroids=simplicial_centroids[indices],
        simplices=simplices[indices]
    )
    return interpolated_points


def _recursively_densify_points(points, radius, iter_results):
    """
    Internal function wrapped by densify. Uses Delaunay triangulation to derive
    a simplicial complex from a given set of points, calcuulates the centroids
    of each simplex, and adds those to the point set. Repeats this process
    recursively on the new point set until no new points can be created.

    Parameters
    ----------
    points : numpy.ndarray
        array of points representing the current state of the point cloud
    radius : float
        minimum distance between interpolated points and existing points
    iter_results : list
        list of InterpolatedPoints objects representing the results of each
        iteration of the algorithm

    Returns
    -------
    synthetic_points : np.ndarray
        set of new interpolated points created by the algorithm
    iter_results : list
        list of InterpolatedPoints objects representing the results of each
        iteration of the algorithm
    """

    simplices = _build_simplicial_complex(points)
    simplices = _sort_simplices_by_volume(simplices)
    new_points = _interpolate_points(points, simplices, radius)

    if not new_points:
        synthetic_points = np.concatenate(
            [pts.centroids for pts in iter_results],
            axis=0
        )
        return synthetic_points, iter_results

    iter_results.append(new_points)

    return _recursively_densify_points(
        np.concatenate((points, new_points.centroids), axis=0),
        radius,
        iter_results
    )


######
# MAIN
######


def densify(points, radius=None):
    """
    Initiates the densify algorithm.

    Parameters
    ----------
    points : numpy.ndarray
        an array of n-dimensional points representing the point cloud to
        "densify"
    radius : float
        minimum distance between new points and existing points

    Returns
    -------
    new_points : numpy.ndarray
        synthetic points created by the algorithm
    iter_results : list
        list of InterpolatedPoints objects representing the results of each
        iteration of the algorithm
    """

    iter_results = []
    new_points, _ = _recursively_densify_points(points, radius, iter_results)

    return new_points, iter_results
