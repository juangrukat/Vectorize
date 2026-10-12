"""Closed cubic least-squares fitting with recursive measured sample error.

Each fit fixes endpoints and estimates two free inner controls by a 2x2
normal system for each coordinate. A fit exceeding tolerance splits at its
largest residual. Corners split runs first, preserving their coordinates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def evaluate_cubic(control: FloatArray, t: FloatArray) -> FloatArray:
    u = 1 - t
    return np.asarray(
        u[:, None] ** 3 * control[0]
        + 3 * (u * u * t)[:, None] * control[1]
        + 3 * (u * t * t)[:, None] * control[2]
        + t[:, None] ** 3 * control[3],
        dtype=np.float64,
    )


def fit_run(points: FloatArray, tolerance: float) -> list[FloatArray]:
    if len(points) < 2:
        return []
    if len(points) == 2:
        delta = (points[1] - points[0]) / 3
        return [np.array([points[0], points[0] + delta, points[1] - delta, points[1]])]
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if cumulative[-1] == 0:
        return []
    t = cumulative / cumulative[-1]
    u = 1 - t
    b1 = 3 * u * u * t
    b2 = 3 * u * t * t
    chord = points[-1] - points[0]
    baseline = np.array(
        [points[0], points[0] + chord / 3, points[-1] - chord / 3, points[-1]]
    )
    basis = np.column_stack((b1, b2))
    # Solve corrections about a translation-covariant baseline. With only
    # three samples the rank-one system otherwise chooses absolute handles
    # near the origin and creates large unsampled excursions.
    corrections = np.linalg.lstsq(
        basis, points - evaluate_cubic(baseline, t), rcond=None
    )[0]
    control = baseline.copy()
    control[1:3] += corrections
    errors = np.linalg.norm(evaluate_cubic(control, t) - points, axis=1)
    split = int(np.argmax(errors))
    if float(errors[split]) <= tolerance:
        return [control]
    split = max(1, min(len(points) - 2, split))
    return fit_run(points[: split + 1], tolerance) + fit_run(points[split:], tolerance)


def corner_indices(points: FloatArray) -> list[int]:
    """Detect sustained corners using five-step chords, suppress grid stairs."""
    window = min(5, max(1, len(points) // 8))
    previous = points - np.roll(points, window, axis=0)
    following = np.roll(points, -window, axis=0) - points
    denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    cosine = np.divide(
        np.sum(previous * following, axis=1),
        denominator,
        out=np.ones(len(points)),
        where=denominator > 0,
    )
    candidates = np.flatnonzero(cosine < 0.65).tolist()
    selected: list[int] = []
    # Local nonmaximum suppression in turning angle picks one true corner.
    for i in sorted(
        candidates,
        key=lambda j: (float(cosine[j]), float(points[j, 0]), float(points[j, 1])),
    ):
        if all(
            min((i - j) % len(points), (j - i) % len(points)) >= window
            for j in selected
        ):
            selected.append(i)
    return sorted(selected)


def fit_closed(points: FloatArray, tolerance: float) -> list[FloatArray]:
    if len(points) < 3:
        return []
    corners = corner_indices(points)
    # Closed runs require two anchors even for a smooth circle.
    anchors = sorted(set(corners + [0, len(points) // 2]))
    controls = []
    for i, start in enumerate(anchors):
        end = anchors[(i + 1) % len(anchors)]
        indices = (
            list(range(start, end + 1))
            if end > start
            else list(range(start, len(points))) + list(range(end + 1))
        )
        controls.extend(fit_run(points[indices], tolerance))
    return controls


def svg_path(points: FloatArray, tolerance: float) -> str:
    curves = fit_closed(points, tolerance)
    if not curves:
        return ""

    def xy(p: FloatArray) -> str:
        return f"{p[0]:.6f},{p[1]:.6f}"

    return (
        "M"
        + xy(curves[0][0])
        + "".join(" C" + xy(c[1]) + " " + xy(c[2]) + " " + xy(c[3]) for c in curves)
        + " Z"
    )


def mesh_paths(
    vertices: FloatArray,
    loops: list[list[int]],
    pinned: NDArray[np.bool_],
    tolerance: float,
) -> list[str]:
    """Cache canonical interface fits; reversed neighbors use identical cubics."""
    cache: dict[tuple[int, ...], list[FloatArray]] = {}
    neighbors: list[set[int]] = [set() for _ in vertices]
    anchor_vertices: set[int] = set()
    for loop in loops:
        for i, vertex in enumerate(loop):
            neighbors[vertex].update((loop[i - 1], loop[(i + 1) % len(loop)]))
        anchor_vertices.update(loop[i] for i in corner_indices(vertices[loop]))
    anchor_vertices.update(
        i for i, adjacent in enumerate(neighbors) if len(adjacent) != 2
    )
    results = []
    for loop in loops:
        points = vertices[loop]
        anchors = [i for i, vertex in enumerate(loop) if vertex in anchor_vertices]
        if len(anchors) < 2:
            # Stable anchors are independent of traversal start and direction.
            first = int(np.argmin(loop))
            opposite = int(np.argmax(np.sum((points - points[first]) ** 2, axis=1)))
            anchors = sorted(set(anchors + [first, opposite]))
        curves = []
        for i, start in enumerate(anchors):
            end = anchors[(i + 1) % len(anchors)]
            indices = (
                list(range(start, end + 1))
                if end > start
                else list(range(start, len(loop))) + list(range(end + 1))
            )
            ids = tuple(loop[j] for j in indices)
            reversed_ids = tuple(reversed(ids))
            key = min(ids, reversed_ids)
            if key not in cache:
                cache[key] = fit_run(vertices[list(key)], tolerance)
            controls = cache[key]
            curves.extend(
                controls if ids == key else [c[::-1] for c in reversed(controls)]
            )

        def xy(p: FloatArray) -> str:
            return f"{p[0]:.6f},{p[1]:.6f}"

        results.append(
            "M"
            + xy(curves[0][0])
            + "".join(" C" + xy(c[1]) + " " + xy(c[2]) + " " + xy(c[3]) for c in curves)
            + " Z"
            if curves
            else ""
        )
    return results
