"""Directed cell edges form signed loops sharing one coordinate mesh.

Positive shoelace winding in screen coordinates is an outer boundary. Holes
have negative winding. Shared coordinates let coverage optimization maintain
coincident interfaces between adjacent colors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class Mesh:
    vertices: FloatArray
    loops: list[list[int]]
    colors: list[int]
    pinned: NDArray[np.bool_]


def signed_area(points: FloatArray) -> float:
    return float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
        / 2
    )


def extract_mesh(labels: IntArray, max_regions: int) -> Mesh:
    h, w = labels.shape
    coordinates: list[tuple[int, int]] = []
    index: dict[tuple[int, int], int] = {}
    loops: list[list[int]] = []
    colors: list[int] = []
    region_count = 0
    for color in np.unique(labels):
        components, count = ndimage.label(labels == color)
        region_count += int(count)
        if region_count > max_regions:
            raise ValueError(
                f"Image has more than {max_regions} connected regions; "
                "reduce detail or colors"
            )
        for region in range(1, int(count) + 1):
            ys, xs = np.nonzero(components == region)
            edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
                for a, b in (
                    ((x, y), (x + 1, y)),
                    ((x + 1, y), (x + 1, y + 1)),
                    ((x + 1, y + 1), (x, y + 1)),
                    ((x, y + 1), (x, y)),
                ):
                    if (b, a) in edges:
                        edges.remove((b, a))
                    else:
                        edges.add((a, b))
            outgoing: dict[tuple[int, int], set[tuple[int, int]]] = {}
            for a, b in edges:
                outgoing.setdefault(a, set()).add(b)
            while edges:
                start, nxt = min(edges)
                points = [start]
                previous, current = start, nxt
                edges.remove((previous, current))
                outgoing[previous].remove(current)
                while current != start:
                    points.append(current)
                    candidates = outgoing[current]
                    dx, dy = current[0] - previous[0], current[1] - previous[1]

                    # Right turn first separates diagonally touching holes.
                    def turn(
                        p: tuple[int, int],
                        current: tuple[int, int] = current,
                        dx: int = dx,
                        dy: int = dy,
                    ) -> tuple[int, tuple[int, int]]:
                        ex, ey = p[0] - current[0], p[1] - current[1]
                        cross = dx * ey - dy * ex
                        dot = dx * ex + dy * ey
                        return (
                            (
                                {1: 0, 0: 1, -1: 2}.get(cross, 3)
                                if dot >= 0 or cross
                                else 3
                            ),
                            p,
                        )

                    nxt = min(candidates, key=turn)
                    edges.remove((current, nxt))
                    outgoing[current].remove(nxt)
                    previous, current = current, nxt
                ids = []
                for point in points:
                    if point not in index:
                        index[point] = len(coordinates)
                        coordinates.append(point)
                    ids.append(index[point])
                loops.append(ids)
                colors.append(int(color))
    vertices = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    neighbors: list[set[int]] = [set() for _ in coordinates]
    for loop in loops:
        for i, v in enumerate(loop):
            neighbors[v].update((loop[i - 1], loop[(i + 1) % len(loop)]))
    pinned = np.asarray(
        [
            x == 0 or y == 0 or x == w or y == h or len(neighbors[i]) != 2
            for i, (x, y) in enumerate(coordinates)
        ],
        dtype=np.bool_,
    )
    # Pin sustained orthogonal corners, while letting one-pixel staircase turns
    # move under reconstruction loss. Three collinear edges distinguish them.
    for loop in loops:
        if len(loop) < 6:
            continue
        corner_points = vertices[loop]
        steps = np.roll(corner_points, -1, axis=0) - corner_points
        for i, vertex in enumerate(loop):
            incoming = steps[(i - 1) % len(loop)]
            outgoing_step = steps[i]
            if float(np.dot(incoming, outgoing_step)) != 0:
                continue
            if all(
                np.array_equal(steps[(i - j) % len(loop)], incoming) for j in (1, 2, 3)
            ) and all(
                np.array_equal(steps[(i + j) % len(loop)], outgoing_step)
                for j in (0, 1, 2)
            ):
                pinned[vertex] = True
    return Mesh(vertices, loops, colors, pinned)
