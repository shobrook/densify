# Standard Library
from itertools import combinations
from collections import namedtuple, defaultdict

# Third Party
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

SyntheticPoints = namedtuple("SyntheticPoints", "centroids simplices")


#########
# HELPERS
#########


def _is_point_left_of_segment(segment, point):
    """
    Tests if a point is on the left or right of a line.

    Parameters
    ----------
    segment : numpy.ndarray
        line segment defined by an initial and final point
    point : numpy.ndarray
        point to test

    Returns
    -------
    is_left : bool
        whether or not the point is on the left of the line
    """

    p0, p1 = segment
    result = (p1[0] - p0[0]) * (point[1] - p0[1])
    result -= (point[0] - p0[0]) * (p1[1] - p0[1])

    if result > 0:
        return True

    return False


def _is_diagonal_outside_polygon(diagonal, vertices):
    """
    Tests if a diagonal –– a line segment conecting two non-adjacent vertices ––
    is within a given polygon. This works by applying the winding number
    algorithm to test if the midpoint of the diagonal is within the polygon.

    Parameters
    ----------
    diagonal : numpy.ndarray
        array of two points representing a line
    vertices : numpy.ndarray
        array of vertices representing a polygon

    Returns
    -------
    is_outside_of_polygon : bool
        boolean that indicates whether the diagonal is outside the polygon
    """

    midpoint = (diagonal[0] + diagonal[1]) / 2
    winding_num = 0
    vertices = tuple(vertices[:]) + (vertices[0], )
    for i in range(len(vertices) - 1):
        v0, v1 = vertices[i], vertices[i + 1]
        edge = np.array([v0, v1])

        if v0[1] <= midpoint[1]:
            if v1[1] > midpoint[1]:
                if _is_point_left_of_segment(edge, midpoint):
                    winding_num += 1
        else:
            if v1[1] <= midpoint[1]:
                if not _is_point_left_of_segment(edge, midpoint):
                    winding_num -= 1

    return winding_num == 0


def _remove_simplices_outside_of_polygon(simplices, vertices):
    """
    Removes simplices that contain edges outside a given polygon.

    Parameters
    ----------
    simplices : numpy.ndarray
        array of simplices representing a triangulation, defined by their
        vertices; shape of [num_simplices, num_vertices, n]
    vertices : numpy.ndarray
        array of points representing the vertices of a polygon, in
        counter-clockwise order

    Returns
    -------
    simplices : numpy.ndarray
        arrary of filtered simplices
    """

    all_simplices = set(range(len(simplices)))
    bad_simplices = set()
    for vertex in vertices:
        for simplex_i in all_simplices: # TODO: Memoize
            # Simplex has already been processed
            if simplex_i in bad_simplices:
                continue

            # Check if each segment in the simplex is within the hull
            for segment in combinations(simplices[simplex_i], 2):
                # TODO: Only check the diagonals (i.e. connecting two vertices)
                if _is_diagonal_outside_polygon(segment, vertices):
                    break
            else:
                continue

            bad_simplices.add(simplex_i)

    all_simplices = set(range(len(simplices)))
    good_simplices = list(all_simplices - bad_simplices)

    return simplices[good_simplices]


def _build_triangulation(points, hull):
    """
    Takes a set of points in n-dimensional space and computes a Delaunay
    triangulation, i.e. a simplicial complex that covers the convex hull of the
    set.

    Parameters
    ----------
    points : numpy.ndarray
        set of points of shape [num_points, n]
    hull : numpy.ndarray
        set of vertices defining the point cloud hull

    Returns
    -------
    simplices : numpy.ndarray
        array of simplices representing the triangulation, defined by their
        vertices; shape of [num_simplices, num_vertices, n]
    """

    triangulation = Delaunay(points)
    simplices = points[triangulation.simplices]

    if hull is None:
        return simplices

    return _remove_simplices_outside_of_polygon(simplices, hull)


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

    interpolated_points = SyntheticPoints(
        centroids=simplicial_centroids[indices],
        simplices=simplices[indices]
    )
    return interpolated_points


def _recursively_densify_points(points, radius, exterior_hull, iter_results):
    """
    Internal function wrapped by densify. Computes a Delaunay triangulation of
    the given set of points, calculates the centroids of each simplex in the
    triangulation, and adds them to the point set. Repeats this process
    recursively on the new point set until no new points can be created.

    Parameters
    ----------
    points : numpy.ndarray
        array of points representing the current state of the point cloud
    radius : float
        minimum distance between interpolated points and existing points
    exterior : numpy.ndarray
        an array of line segments defining the exterior hull of the point cloud,
        used to perform a constrained Delaunay triangulation on a non-convex
        set; no points can be outside this hull
    iter_results : list
        list of SyntheticPoints objects representing the results of each
        iteration of the algorithm

    Returns
    -------
    synthetic_points : np.ndarray
        set of new interpolated points created by the algorithm
    iter_results : list
        list of SyntheticPoints objects representing the results of each
        iteration of the algorithm
    """

    simplices = _build_triangulation(points, exterior_hull)
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
        exterior_hull,
        iter_results
    )


######
# MAIN
######


def densify(points, radius=None, exterior_hull=None):
    """
    Initiates the densify algorithm.

    Parameters
    ----------
    points : numpy.ndarray
        an array of n-dimensional points representing the point cloud to
        "densify"
    radius : float
        minimum distance between new points and existing points
    exterior : numpy.ndarry
        an array of vertices defining the exterior hull of the point cloud, used
        to perform a constrained Delaunay triangulation on a non-convex set; no
        points can be outside this hull

    Returns
    -------
    new_points : numpy.ndarray
        synthetic points created by the algorithm
    iter_results : list
        list of SyntheticPoints objects representing the results of each
        iteration of the algorithm
    """

    iter_results = []
    new_points, _ = _recursively_densify_points(points, radius, exterior_hull,
                                                iter_results)

    return new_points, iter_results
