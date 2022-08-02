# densify

`densify` is an algorithm for [oversampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) point clouds. It creates synthetic data points to "fill the gaps" in the cloud, making the cloud more dense. This is a useful [technique for reducing overfitting](https://en.wikipedia.org/wiki/Regularization_(mathematics)) when training machine learning models on point cloud datasets.

TODO: Animated demo

## Installation

You can install `densify` from PyPi:

```bash
$ pip install densify
```

## Usage

Using `densify` is very simple. The function expects a numpy array representing your point cloud (shape == `[num_points, point_dim]`) and a "radius" that dictates the minimum distance each synthetic point must be from all other points. You can think of radius as a measure of density (smaller => more dense).

```python
from densify import densify

point_cloud = np.array([[4.6, 6.5],
                        [1.5, 4.1],
                        [6.1, 5.0],
                        [1.1, 2.9],
                        [10.0, 5.0]])
new_points, _ = densify(point_cloud, radius=0.25)
```

The function returns the synthetic points as a numpy array and a list of algorithm outputs to plug into `visualize_densify`.

TODO: Visualizing algorithm results

## How it Works

`densify` computes a [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of the point cloud and creates synthetic points from the centroids of each simplex in the triangulation. These points are added to the cloud, and the process is repeated recursively until no new points can be created.
