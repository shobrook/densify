# densify

`densify` is an algorithm for [oversampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) point clouds. It creates synthetic data points that "fill the gaps" in the cloud, making it more dense. This can be a useful [technique for reducing overfitting](https://en.wikipedia.org/wiki/Regularization_(mathematics)) when training machine learning models on point cloud datasets.

<!--TODO: Demo-->

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
new_points, _ = densify(point_cloud, radius=0.15)
```

The function returns the synthetic points as a numpy array and a list of algorithm outputs to plug into `visualize_densify`.

<!--TODO: How to visualize algorithm results-->

## How it Works

`densify` computes a [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of the point cloud and creates synthetic points from the centroids of each simplex in the triangulation. These points are added to the cloud, and the process is repeated recursively until no new points can be created.
