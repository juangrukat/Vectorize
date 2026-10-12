import numpy as np

from vector_server.curves import (
    evaluate_cubic,
    fit_closed,
    fit_run,
    mesh_paths,
    svg_path,
)
from vector_server.geometry import extract_mesh


def test_line_is_exact():
    points = np.column_stack((np.linspace(0, 10, 20), np.linspace(0, 10, 20)))
    curves = fit_run(points, 0.001)
    assert len(curves) == 1
    np.testing.assert_allclose(
        evaluate_cubic(curves[0], np.linspace(0, 1, 20)), points, atol=1e-10
    )
    assert fit_run(points[:1], 0.1) == []
    assert fit_run(np.zeros((4, 2)), 0.1) == []


def test_square_corners_and_closure():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    curves = fit_closed(square, 0.1)
    assert len(curves) == 4
    for i, curve in enumerate(curves):
        np.testing.assert_allclose(curve[-1], curves[(i + 1) % 4][0])
    assert svg_path(square, 0.1).endswith(" Z")
    assert svg_path(square[:2], 0.1) == ""


def test_recursive_fit_circle():
    t = np.linspace(0, np.pi, 31)
    points = np.column_stack((np.cos(t), np.sin(t)))
    curves = fit_run(points, 0.002)
    assert len(curves) > 1
    for c in curves:
        assert np.isfinite(c).all()


def test_shared_curve_reverse():
    labels = np.zeros((12, 12), dtype=np.int64)
    labels[3:9, 3:9] = 1
    mesh = extract_mesh(labels, 10)
    paths = mesh_paths(mesh.vertices, mesh.loops, mesh.pinned, 0.1)
    assert len(paths) == len(mesh.loops)
    assert all(p.endswith(" Z") for p in paths)


def test_smooth_shared_ring_uses_identical_controls():
    import re

    from vector_server.geometry import signed_area

    labels = np.zeros((12, 12), dtype=np.int64)
    labels[3:9, 3:9] = 1
    mesh = extract_mesh(labels, 10)
    shared = [
        i
        for i, loop in enumerate(mesh.loops)
        if abs(signed_area(mesh.vertices[loop])) == 36
    ]
    # Displace shared boundary vertices so fits have nontrivial controls.
    mesh.vertices[:, 0] += 0.1 * np.sin(mesh.vertices[:, 1])
    paths = mesh_paths(mesh.vertices, mesh.loops, mesh.pinned, 0.1)

    def controls(path):
        numbers = np.asarray(
            [float(v) for v in re.findall(r"-?\d+\.\d+", path)]
        ).reshape(-1, 2)
        curves = []
        start = numbers[0]
        for i in range(1, len(numbers), 3):
            curve = np.vstack([start, numbers[i : i + 3]])
            curves.append(curve)
            start = curve[-1]
        return curves

    first = controls(paths[shared[0]])
    second = controls(paths[shared[1]])
    normalized_a = sorted(tuple(c.ravel()) for c in first)
    normalized_b = sorted(tuple(c[::-1].ravel()) for c in second)
    assert normalized_a == normalized_b


def test_rank_deficient_fit_is_translation_covariant_and_bounded():
    points = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]])
    translated = points + np.array([100.0, 100.0])
    first = fit_run(points, 0.001)
    shifted = fit_run(translated, 0.001)
    assert len(first) == len(shifted) == 1
    np.testing.assert_allclose(shifted[0], first[0] + [100.0, 100.0], atol=1e-10)
    dense = evaluate_cubic(shifted[0], np.linspace(0, 1, 1001))
    np.testing.assert_allclose(dense[:, 0], 100.0, atol=1e-10)
    assert dense[:, 1].min() >= 100.0 - 1e-10
    assert dense[:, 1].max() <= 101.0 + 1e-10


def test_fit_translation_covariance_general_runs():
    for size in (2, 3, 4, 12):
        t = np.linspace(0, 1, size)
        points = np.column_stack((t, t * t))
        first = fit_run(points, 0.005)
        shifted = fit_run(points + [300.0, -200.0], 0.005)
        assert len(first) == len(shifted)
        for original, moved in zip(first, shifted, strict=True):
            np.testing.assert_allclose(moved, original + [300.0, -200.0], atol=1e-9)
