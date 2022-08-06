import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from itertools import combinations
from dataclasses import dataclass

EDGE_FRAME_MULTIPLIER = 50 # No. of edge frames per densify iteration
NODE_FRAME_MULTIPLIER = 5 # No. of frames to show for the addition of a point


#####################
# POINT CLOUD HELPERS
#####################


def create_final_point_cloud(init_points, iter_results):
    final_points = [init_points] + [r.centroids for r in iter_results]
    final_points = np.concatenate(final_points)

    return final_points


def point_cloud_to_graph(final_points, iter_results):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(final_points)))

    return graph


##################
# ANIMATION FRAMES
##################


@dataclass
class NodeFrame:
    node_indices: np.ndarray


@dataclass
class EdgeFrame:
    edges: np.ndarray
    percentage: float = 0.0


def generate_edge_frames(simplices):
    for i in range(EDGE_FRAME_MULTIPLIER):
        for j in range(simplices.shape[0]):
            vertices = simplices[j]
            edges = [np.array(edge) for edge in combinations(vertices, 2)]

            yield EdgeFrame(edges, percentage=i / EDGE_FRAME_MULTIPLIER)


def generate_node_frames(curr_points, prior_points, final_points):
    node_indices = np.argwhere(np.isin(curr_points, final_points).all(axis=1))
    for node_index in node_indices:
        prior_points = np.concatenate(curr_indices, node_index)

        for _ in range(NODE_FRAME_MULTIPLIER):
            yield NodeFrame(prior_points)


def generate_frames(init_points, final_points, iter_results):
    prior_points = np.argwhere(np.isin(curr_points, final_points).all(axis=1))
    frames = []
    for result in iter_results:
        edge_frames = generate_edge_frames(result.simplices)
        node_frames = generate_node_frames(result.centroids, prior_points, final_points)

        frames.extend(edge_frames)
        frames.extend(node_frames)

        centroid_indices = np.argwhere(np.isin(result.centroids, final_points).all(axis=1))
        prior_points = np.concatenate(prior_points, centroid_indices)

    return frames


######
# MAIN
######


class Animation(object):
    def __init__(self, init_points, iter_results, dark=True, seed=2):
        np.random.seed(seed)

        self.init_points = init_points # Needed?
        self.final_points = create_final_point_cloud(init_points, iter_results) # Does it need to be an instance variable?
        self.final_graph = point_cloud_to_graph(self.final_points, iter_results)

        self.is_dark = dark
        plt.rcParams["figure.facecolor"] = "black" if dark else "white"
        plt.rcParams["axes.facecolor"] = "black" if dark else "white"

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.artists = None

        # Animation frames
        self.frames = generate_frames(init_points, final_points, iter_results)

    def init_fig(self):
        pass

    def update(self, i):
        frame = self.frames[i]
        pass

    def show(self, duration=15, filename=None, dpi=None):
        pass
