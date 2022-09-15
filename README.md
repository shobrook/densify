# densify

`densify` is an algorithm for [oversampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) point clouds. It creates synthetic data points that "fill the gaps" in the cloud, making it more dense. This can be a useful [technique for reducing overfitting](https://en.wikipedia.org/wiki/Regularization_(mathematics)) when training machine learning models on point cloud datasets.

![Demo](demo.gif)

## Installation

You can install `densify` from PyPi:

```bash
$ pip install densify
```

## Usage

`densify` is simple to use. The function expects an array of points representing your cloud and a "radius" that dictates the minimum distance each synthetic point must be from all other points. The smaller the radius, the higher the density.

```python
import numpy as np
from densify import densify

point_cloud = np.array([[4.6, 6.5],
                        [1.5, 4.1],
                        [6.1, 5.0],
                        [1.1, 2.9],
                        [10.0, 5.0]])
new_points, iter_results = densify(point_cloud, radius=0.15)
```

The function returns `new_points`, a numpy array of the synthetic points, and `iter_results`, a list of algorithm outputs to plug into `visualize_densify`.

### Constrained Point Generation

By default, `densify` acts within the [convex hull](https://en.wikipedia.org/wiki/Convex_hull) of the point cloud and will not create points outside that boundary. But if the point cloud is non-convex, you can define a boundary to generate points within. To do this, pass in a list of points in the cloud representing the boundary:

```python
point_cloud = np.array([[0, 0],
                        [4, 0],
                        [4, -3],
                        [6, -3],
                        [6, 3],
                        [3, 5],
                        [2, 1],
                        [3, 3],
                        [5, 0],
                        [4, 1]])
hull = np.array([[0, 0],
                 [4, 0],
                 [4, -3],
                 [6, -3],
                 [6, 3],
                 [3, 5]])
new_points, iter_results = densify(point_cloud, radis=0.15, exterior_hull=hull)
```

Note that these points must define a _simple_ polygon that encloses _all_ the points in the cloud.

### Visualizing Point Generation

`densify` lets you visualize the point generation process for 2D point clouds. Simply plug the `point_cloud` and `iter_results` objects into `animate_densify`:

```python
from densify import animate_densify

animate_densify(point_cloud, iter_results, dark=True, filename="ani.gif")
```

## How it Works

`densify` computes a [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of the point cloud and creates synthetic points from the centroids of each simplex in the triangulation. These points are added to the cloud, and the process is repeated recursively until no new points can be created.

If a boundary is given, `densify` enforces it by using the [winding number algorithm](https://en.wikipedia.org/wiki/Point_in_polygon#Winding_number_algorithm) to identify simplices that contain edges outside of the boundary, and then dropping them.

# Authors

`densify` was created by Jonathan Shobrook with the help of [Paul C. Bogdan](https://github.com/paulcbogdan/) as part of our research in the [Dolcos Lab](https://dolcoslab.beckman.illinois.edu/) at the Beckman Institute for Advanced Science and Technology.
