import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from itertools import combinations
from dataclasses import dataclass
from typing import Dict
from collections import namedtuple

Artists = namedtuple("Artists", "node_paths edge_segments")

INIT_FIG_MULTIPLIER = 50
EDGE_FRAME_MULTIPLIER = 50 # No. of edge frames per densify iteration
NODE_FRAME_MULTIPLIER = 1 # No. of frames to show for the addition of a point

DARK_NODE_COLOR = "white"
LIGHT_NODE_COLOR = (1.0, 1.0, 1.0, 0.75)
GREY = (1.0, 1.0, 1.0, 0.25)
LIGHT_GREY = (0.6, 0.6, 0.6, 1.0)


#####################
# POINT CLOUD HELPERS
#####################


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
            # TODO: Separate argwhere call into its own function
            vertices = np.argwhere(np.logical_not(np.isin(final_points, simplex, invert=True).any(axis=1))).squeeze()
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
class NodeFrame:
    nodelist: np.ndarray


@dataclass
class EdgeFrame:
    edge_segments: np.ndarray
    percentage: float = 0.0


def generate_edge_frames(simplices):
    segments = generate_edge_segments(simplices)
    for i in range(EDGE_FRAME_MULTIPLIER):
        yield EdgeFrame(segments, percentage=i / EDGE_FRAME_MULTIPLIER)


def generate_node_frames(curr_points, prior_points, final_points):
    node_indices = np.argwhere(np.isin(final_points, curr_points).all(axis=1))
    prior_points = prior_points.squeeze()
    for node_index in node_indices:
        prior_points = np.concatenate([prior_points, node_index])

        for _ in range(NODE_FRAME_MULTIPLIER):
            yield NodeFrame(prior_points)


def generate_frames(init_points, final_points, iter_results):
    prior_points = np.argwhere(np.isin(init_points, final_points).all(axis=1))
    frames = []
    for result in iter_results:
        edge_frames = generate_edge_frames(result.simplices)
        node_frames = generate_node_frames(result.centroids, prior_points, final_points)

        frames.extend(edge_frames)
        frames.extend(node_frames)

        centroid_indices = np.argwhere(np.isin(result.centroids, final_points).all(axis=1))
        prior_points = np.concatenate([prior_points, centroid_indices])

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

        self.fig, self.ax0 = plt.subplots(figsize=(12, 6))
        self.artists = None

        # Animation frames
        self.frames = generate_frames(init_points, self.final_points, iter_results)

    def init_fig(self):
        for spine in self.ax0.spines.values():
            spine.set_visible(False)

        init_nodes = np.argwhere(np.isin(self.init_points, self.final_points).all(axis=1)).squeeze()
        node_paths = nx.draw_networkx_nodes(
            self.final_graph,
            pos=nx.get_node_attributes(self.final_graph, "pos"),
            node_color=DARK_NODE_COLOR if self.is_dark else LIGHT_GREY,
            node_size=15,
            nodelist=init_nodes,
            # linewidths=34 / len(self.final_points), # Pull out into variable/method
            ax=self.ax0
        )
        edge_segments = nx.draw_networkx_edges(
            self.final_graph,
            pos=nx.get_node_attributes(self.final_graph, "pos"),
            edge_color="white" if self.is_dark else LIGHT_GREY, # GREY
            # ax=self.ax0
            width=34 / len(self.final_points)
            # edgelist=[]
        )
        self.artists = Artists(node_paths, edge_segments)
        self.artists.edge_segments.set_segments([])

        return self.artists

    def update(self, i):
        if i < INIT_FIG_MULTIPLIER:
            return self.artists
        else:
            i -= INIT_FIG_MULTIPLIER

        frame = self.frames[i]

        if isinstance(frame, EdgeFrame):
            segments = generate_partial_edge_segments(frame.edge_segments, frame.percentage)
            self.artists.edge_segments.set_segments(segments)
        elif isinstance(frame, NodeFrame):
            self.artists = Artists(
                nx.draw_networkx_nodes(
                    self.final_graph,
                    pos=nx.get_node_attributes(self.final_graph, "pos"),
                    node_color=DARK_NODE_COLOR if self.is_dark else LIGHT_GREY,
                    node_size=15,
                    nodelist=frame.nodelist,
                    # linewidths=34 / len(self.final_points), # Pull out into variable/method
                    ax=self.ax0
                ),
                self.artists.edge_segments
            )

        return self.artists

    def show(self, duration=15, filename=None, dpi=None):
        num_frames = len(self.frames) + INIT_FIG_MULTIPLIER
        interval = (1000 * duration) / num_frames
        fps = num_frames / duration

        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=num_frames,
            init_func=self.init_fig,
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
    new_points, iter_results = densify(points, radius=0.25)

    anim = Animation(points, iter_results)
    anim.show()
