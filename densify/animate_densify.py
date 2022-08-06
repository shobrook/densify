import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from itertools import combinations
from dataclasses import dataclass
from collections import namedtuple

Artists = namedtuple("Artists", "node_paths edge_segments")

EDGE_FRAME_MULTIPLIER = 50 # No. of edge frames per densify iteration
NODE_FRAME_MULTIPLIER = 5 # No. of frames to show for the addition of a point

GREY = (1.0, 1.0, 1.0, 0.75)
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
    graph.add_nodes_from(range(len(final_points)))

    # TODO: Clean up and use add_edges_from
    for r in iter_results:
        for simplex in r.simplices:
            vertices = np.argwhere(np.logical_not(np.isin(final_points, simplex, invert=True).any(axis=1))).squeeze()
            for edge in combinations(vertices, 2):
                graph.add_edge(*edge)

    return graph


##################
# ANIMATION FRAMES
##################


@dataclass
class NodeFrame:
    node_paths: np.ndarray


@dataclass
class EdgeFrame:
    edge_segments: np.ndarray
    percentage: float = 0.0


def generate_edge_frames(simplices):
    segments = []
    for i in range(simplices.shape[0]):
        vertices = simplices[i]
        simplex_segments = (np.array(seg) for seg in combinations(vertices, 2))
        segments.extend(simplex_segments)

    for i in range(EDGE_FRAME_MULTIPLIER):
        yield EdgeFrame(segments, percentage=i / EDGE_FRAME_MULTIPLIER)


def generate_node_frames(curr_points, prior_points, final_points):
    node_indices = np.argwhere(np.isin(curr_points, final_points).all(axis=1))
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


#
#
#


def generate_edge_segments(segments, percentage):
    for segment in segments:
        src, targ = segment[0], segment[1]
        midpoint = (src + targ) / 2

        partial_src = midpoint + ((src - midpoint) * percentage)
        partial_targ = midpoint + ((targ - midpoint) * percentage)
        partial_segment = np.array([partial_src, partial_targ])

        yield partial_segment


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

        # self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig, self.ax0 = plt.subplots()
        self.artists = None

        self.pos = {i: self.final_points[i] for i in range(len(self.final_points))}

        # Animation frames
        self.frames = generate_frames(init_points, self.final_points, iter_results)

    def init_fig(self):
        for spine in self.ax0.spines.values():
            spine.set_visible(False)

        # TODO: Probably a better way to do this; move into function that creates node paths for given points
        nodes_to_keep = np.argwhere(np.isin(self.init_points, self.final_points).all(axis=1)).squeeze()
        nodes_to_remove = np.argwhere(np.isin(self.final_points, self.init_points, invert=True).any(axis=1)).squeeze()

        init_graph = self.final_graph.copy()
        init_graph.remove_nodes_from(nodes_to_remove)

        node_paths = nx.draw_networkx_nodes(
            init_graph,
            pos={i: self.final_points[i] for i in nodes_to_keep},
            node_color="white" if self.is_dark else LIGHT_GREY,
            node_size=15,
            # linewidths=34 / len(self.final_points), # Pull out into variable/method
            ax=self.ax0
        )
        edge_segments = nx.draw_networkx_edges(
            self.final_graph,
            pos=self.pos,
            edge_color="white" if self.is_dark else LIGHT_GREY, # GREY
            # ax=self.ax0
            # width=34 / len(self.final_points)
        )
        self.artists = Artists(node_paths, edge_segments)

        return self.artists

    def update(self, i):
        frame = self.frames[i]
        if isinstance(frame, EdgeFrame):
            segments = generate_edge_segments(frame.edge_segments, frame.percentage)
            self.artists.edge_segments.set_segments(segments)
        elif isinstance(frame, NodeFrame):
            pass
            # self.artists.node_paths = frame.node_paths

        return self.artists

    def show(self, duration=15, filename=None, dpi=None):
        fps = int(len(self.frames) / duration)
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.frames),
            init_func=self.init_fig,
            interval=1000 / fps if not filename else 200,
            # save_count
            blit=False
        )
        plt.show()


if __name__ == "__main__":
    from densify import densify

    points = np.array([[4.6, 6.5],
                       [1.5, 4.1],
                       [6.1, 5.0],
                       [1.1, 2.9],
                       [10.0, 5.0]])
    new_points, iter_results = densify(points, radius=0.5)

    anim = Animation(points, iter_results)
    anim.show()
