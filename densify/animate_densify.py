import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from itertools import combinations
from dataclasses import dataclass
from typing import Dict
from collections import namedtuple
from math import ceil

Artists = namedtuple("Artists", "node_paths edge_segments")

INIT_FRAME_MULTIPLIER = 50
EDGE_FRAME_MULTIPLIER = 50 # No. of edge frames per densify iteration
NODE_FRAME_MULTIPLIER = 25 # No. of frames to show for the addition of a point
FINAL_FRAME_MULTIPLIER = 50

DARK_NODE_COLOR = "white"
LIGHT_NODE_COLOR = (1.0, 1.0, 1.0, 0.75)
DARK_EDGE_COLOR = (0.6, 0.6, 0.6, 1.0)
LIGHT_EDGE_COLOR = (1.0, 1.0, 1.0, 0.5)


#####################
# POINT CLOUD HELPERS
#####################


def points_to_nodes(points, final_points):
    return np.atleast_1d(np.argwhere(np.isin(final_points, points).all(axis=1)).squeeze())


def create_final_point_cloud(init_points, iter_results):
    final_points = [init_points] + [r.centroids for r in iter_results]
    final_points = np.concatenate(final_points)

    return final_points


def point_cloud_to_graph(final_points, iter_results):
    graph = nx.Graph()
    for node, pos in enumerate(final_points):
        graph.add_node(node, pos=pos)

    # TODO: Clean up and use add_edges_from
    for r in iter_results:
        for simplex in r.simplices:
            vertices = points_to_nodes(simplex, final_points)
            for edge in combinations(vertices, 2):
                graph.add_edge(*edge)

    return graph


######################
# EDGE SEGMENT HELPERS
######################


def generate_edge_segments(simplices):
    segments = []
    for i in range(simplices.shape[0]):
        vertices = simplices[i]
        simplex_segments = (np.array(seg) for seg in combinations(vertices, 2))

        segments.extend(simplex_segments)

    return segments


def generate_partial_edge_segments(segments, percentage):
    for segment in segments:
        src, targ = segment[0], segment[1]
        midpoint = (src + targ) / 2

        partial_src = midpoint + ((src - midpoint) * percentage)
        partial_targ = midpoint + ((targ - midpoint) * percentage)
        partial_segment = np.array([partial_src, partial_targ])

        yield partial_segment


##################
# ANIMATION FRAMES
##################


@dataclass
class Frame:
    nodelist: np.ndarray = np.array([])
    edge_segments: np.ndarray = np.array([])
    edge_percentage: float = 0.0


def generate_edge_frames(edge_segments, prior_nodes):
    for i in range(EDGE_FRAME_MULTIPLIER):
        edge_percentage = i / EDGE_FRAME_MULTIPLIER
        yield Frame(prior_nodes, edge_segments, edge_percentage)


def generate_node_frames(curr_nodes, prior_nodes, edge_segments):
    for node in curr_nodes:
        prior_nodes = np.concatenate([prior_nodes, np.atleast_1d(node)])

        num_frames = min(NODE_FRAME_MULTIPLIER, ceil(NODE_FRAME_MULTIPLIER / len(curr_nodes)))
        for _ in range(num_frames):
            yield Frame(prior_nodes, edge_segments, edge_percentage=1.0)


def generate_frames(init_nodes, final_points, iter_results):
    for _ in range(INIT_FRAME_MULTIPLIER):
        yield Frame(init_nodes)

    prior_nodes = init_nodes
    for result in iter_results:
        curr_nodes = points_to_nodes(result.centroids, final_points)
        edge_segments = generate_edge_segments(result.simplices)

        yield from generate_edge_frames(edge_segments, prior_nodes)
        yield from generate_node_frames(curr_nodes, prior_nodes, edge_segments)

        prior_nodes = np.concatenate([prior_nodes, curr_nodes])

    for _ in range(FINAL_FRAME_MULTIPLIER):
        yield Frame(np.arange(final_points.shape[0]))


######
# MAIN
######


class Animation(object):
    def __init__(self, init_points, iter_results, dark=True, seed=2):
        np.random.seed(seed)

        final_points = create_final_point_cloud(init_points, iter_results)
        self.final_graph = point_cloud_to_graph(final_points, iter_results)
        self.init_nodes = points_to_nodes(init_points, final_points)

        self.is_dark = dark
        plt.rcParams["figure.facecolor"] = "black" if dark else "white"
        plt.rcParams["axes.facecolor"] = "black" if dark else "white"

        self.fig, self.ax0 = plt.subplots(figsize=(12, 6))
        self.artists = None

        # Animation frames
        self.frames = list(generate_frames(self.init_nodes, final_points, iter_results))

    def update(self, i):
        if self.artists:
            self.artists.node_paths.remove()
            self.artists.edge_segments.remove()

        frame = self.frames[i]
        node_paths = nx.draw_networkx_nodes(
            self.final_graph,
            pos=nx.get_node_attributes(self.final_graph, "pos"),
            node_color=DARK_NODE_COLOR if self.is_dark else LIGHT_NODE_COLOR,
            node_size=15,
            nodelist=frame.nodelist,
            ax=self.ax0
        )
        edge_segments = nx.draw_networkx_edges(
            self.final_graph,
            pos=nx.get_node_attributes(self.final_graph, "pos"),
            edge_color=DARK_EDGE_COLOR if self.is_dark else LIGHT_EDGE_COLOR,
            # width=34 / self.final_graph.number_of_nodes()
            # width=15 / len(frame.nodelist)
            width=0.75
        )
        partial_edge_segments = generate_partial_edge_segments(frame.edge_segments, frame.edge_percentage)
        edge_segments.set_segments(partial_edge_segments)

        self.artists = Artists(node_paths, edge_segments)

        return self.artists

    def show(self, duration=15, filename=None, dpi=None):
        num_frames = len(self.frames)
        interval = (1000 * duration) / num_frames
        fps = num_frames / duration

        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=num_frames,
            interval=interval if not filename else fps,
            blit=False,
            repeat=False
        )
        plt.show()


if __name__ == "__main__":
    from densify import densify

    points = np.array([[4.6, 6.5],
                       [1.5, 4.1],
                       [6.1, 5.0],
                       [1.1, 2.9],
                       [10.0, 5.0]])
    new_points, iter_results = densify(points, radius=0.15)

    anim = Animation(points, iter_results)
    anim.show()


    # TODO: Make initial nodes a different color
        # OR a gradient! Start out white and make more red with each iteration
    # TODO: Make a separate graph for density over time
    # TODO: Give a title
    # TODO: Try out Paul's suggestion
