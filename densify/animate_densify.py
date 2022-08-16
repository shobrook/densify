import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.text import Text
from scipy.spatial import ConvexHull

from itertools import combinations
from dataclasses import dataclass
from typing import Dict
from collections import namedtuple
from math import ceil

INIT_FRAME_MULTIPLIER = 50
EDGE_FRAME_MULTIPLIER = 50 # No. of edge frames per densify iteration
NODE_FRAME_MULTIPLIER = 25 # No. of frames to show for the addition of a point
FINAL_FRAME_MULTIPLIER = 50

DARK_TITLE_COLOR = "white"
LIGHT_TITLE_COLOR = (0.3, 0.3, 0.3, 1.0)
DARK_NODE_COLOR = "white"
LIGHT_NODE_COLOR = (0.3, 0.3, 0.3, 1.0)
DARK_EDGE_COLOR = (0.6, 0.6, 0.6, 1.0)
LIGHT_EDGE_COLOR = (0.6, 0.6, 0.6, 1.0)
# RED = (0.69, 0.05, 0.18, 1.0)
RED = "red"
DARK_BACKGROUND = (0.05, 0.05, 0.05, 1.0)
LIGHT_BACKGROUND = "white"


##############
# DATA CLASSES
##############


@dataclass
class Frame:
    nodelist: np.ndarray = np.array([])
    edge_segments: np.ndarray = np.array([])
    edge_percentage: float = 0.0
    iter_num: int = 0
    density: float = 0.0


@dataclass
class Artists:
    node_paths: PathCollection = PathCollection([])
    edge_segments: LineCollection = LineCollection([])
    title: Text = Text()


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


def point_cloud_to_convex_hull(points):
    hull = ConvexHull(points)

    x, y = points[hull.vertices, 0], points[hull.vertices, 1]
    vertices = np.column_stack((x, y))

    return vertices


def area_of_2d_convex_hull(vertices):
    # https://en.wikipedia.org/wiki/Shoelace_formula#Statement
    A = 0.0
    for i in range(-1, vertices.shape[0] - 1):
        A += vertices[i][0] * (vertices[i + 1][1] - vertices[i - 1][1])

    return A / 2


def calculate_point_cloud_density(points, convex_hull):
    area = area_of_2d_convex_hull(convex_hull)
    return round(len(points) / area, 3)


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


def generate_edge_frames(edge_segments, prior_nodes):
    for i in range(EDGE_FRAME_MULTIPLIER):
        edge_percentage = i / EDGE_FRAME_MULTIPLIER
        yield Frame(nodelist=prior_nodes, edge_segments=edge_segments, edge_percentage=edge_percentage)


def generate_node_frames(curr_nodes, prior_nodes, edge_segments):
    for node in curr_nodes:
        prior_nodes = np.concatenate([prior_nodes, np.atleast_1d(node)])

        num_frames = min(NODE_FRAME_MULTIPLIER, ceil(NODE_FRAME_MULTIPLIER / len(curr_nodes)))
        for _ in range(num_frames):
            yield Frame(nodelist=prior_nodes, edge_segments=edge_segments, edge_percentage=1.0)


def set_metadata_for_frames(frames, convex_hull, iter_num):
    for frame in frames:
        frame.iter_num = iter_num + 1
        frame.density = calculate_point_cloud_density(frame.nodelist, convex_hull)

        yield frame


def generate_frames(init_nodes, final_points, iter_results):
    convex_hull = point_cloud_to_convex_hull(final_points[init_nodes])
    init_density = calculate_point_cloud_density(init_nodes, convex_hull)

    for _ in range(INIT_FRAME_MULTIPLIER):
        yield Frame(nodelist=init_nodes, density=init_density)

    prior_nodes = init_nodes
    for iter_num, result in enumerate(iter_results):
        curr_nodes = points_to_nodes(result.centroids, final_points)
        edge_segments = generate_edge_segments(result.simplices)

        edge_frames = generate_edge_frames(edge_segments, prior_nodes)
        node_frames = generate_node_frames(curr_nodes, prior_nodes, edge_segments)

        yield from set_metadata_for_frames(edge_frames, convex_hull, iter_num)
        yield from set_metadata_for_frames(node_frames, convex_hull, iter_num)

        prior_nodes = np.concatenate([prior_nodes, curr_nodes])

    final_density = calculate_point_cloud_density(final_points, convex_hull)

    for _ in range(FINAL_FRAME_MULTIPLIER):
        yield Frame(nodelist=np.arange(final_points.shape[0]), density=final_density)


######
# MAIN
######


class Animation(object):
    def __init__(self, init_points, iter_results, dark=True):
        final_points = create_final_point_cloud(init_points, iter_results)
        self.final_graph = point_cloud_to_graph(final_points, iter_results)
        self.init_nodes = points_to_nodes(init_points, final_points)

        self.is_dark = dark
        plt.rcParams["figure.facecolor"] = DARK_BACKGROUND if dark else LIGHT_BACKGROUND
        plt.rcParams["axes.facecolor"] = DARK_BACKGROUND if dark else LIGHT_BACKGROUND

        self.fig, self.ax0 = plt.subplots(figsize=(12, 6))
        self.artists = None

        # Animation frames
        self.frames = list(generate_frames(self.init_nodes, final_points, iter_results))

    def _get_node_colors(self, frame):
        num_init_nodes = len(self.init_nodes)
        num_total_nodes = len(frame.nodelist)

        node_colors = [DARK_NODE_COLOR if self.is_dark else LIGHT_NODE_COLOR for _ in range(num_init_nodes)]
        node_colors += [RED for _ in range(num_init_nodes, num_total_nodes)]

        return node_colors

    def init_fig(self):
        for spine in self.ax0.spines.values():
            spine.set_visible(False)

    def update(self, i):
        if self.artists:
            self.artists.node_paths.remove()
            self.artists.edge_segments.remove()
            self.artists.title.remove()

        frame = self.frames[i]
        title = self.ax0.text(
            s=f"Iteration #{frame.iter_num} (density={frame.density})",
            transform=self.ax0.transAxes,
            x=0.5,
            y=1.05,
            size=plt.rcParams["axes.titlesize"],
            ha="center",
            color=DARK_TITLE_COLOR if self.is_dark else LIGHT_TITLE_COLOR
        )
        node_paths = nx.draw_networkx_nodes(
            self.final_graph,
            pos=nx.get_node_attributes(self.final_graph, "pos"),
            node_color=self._get_node_colors(frame),
            node_size=15,
            nodelist=frame.nodelist,
            ax=self.ax0
        )
        edge_segments = nx.draw_networkx_edges(
            self.final_graph,
            pos=nx.get_node_attributes(self.final_graph, "pos"),
            edge_color=DARK_EDGE_COLOR if self.is_dark else LIGHT_EDGE_COLOR,
            # width=15 / len(frame.nodelist)
            width=0.75,
            ax=self.ax0
        )
        partial_edge_segments = generate_partial_edge_segments(frame.edge_segments, frame.edge_percentage)
        edge_segments.set_segments(partial_edge_segments)

        self.artists = Artists(node_paths, edge_segments, title)

        return self.artists

    def show(self, duration=15, filename=None, dpi=None):
        num_frames = len(self.frames)
        interval = (1000 * duration) / num_frames
        fps = num_frames / duration

        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init_fig,
            frames=num_frames,
            interval=interval if not filename else fps,
            blit=False,
            repeat=False
        )

        if not filename:
            plt.show()
        else:
            anim.save(filename, fps=fps, dpi=dpi)

        return anim


def animate_densify(init_points, iter_results, dark=True, duration=15,
                    filename=None, dpi=None):
    anim = Animation(init_points, iter_results, dark)
    return anim.show(duration, filename, dpi)


if __name__ == "__main__":
    from densify import densify

    points = np.array([[4.6, 6.5],
                       [1.5, 4.1],
                       [6.1, 5.0],
                       [1.1, 2.9],
                       [10.0, 5.0]])
    new_points, iter_results = densify(points, radius=0.15)

    animate_densify(points, iter_results)

    # TODO: Implement Paul's suggestion
    # TODO: Implement a color gradient for the nodes
