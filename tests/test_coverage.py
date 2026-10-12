"""Coverage uses signed shoelace areas after four pixel half-plane clips.

The kernel differentiates every line/clip intersection and then the area;
finite differences here independently verify those analytic derivatives.
"""

import numpy as np
import pytest

from vector_server import _coverage


def test_fractional_rectangle():
    vertices = np.array([[0.25, 0.25], [1.75, 0.25], [1.75, 1.75], [0.25, 1.75]])
    palette = np.array([[0.2, 0.4, 0.6, 1.0]])
    result = _coverage.render(vertices, [[0, 1, 2, 3]], [0], palette, 2, 2)
    np.testing.assert_allclose(result, np.broadcast_to(palette[0] * 0.5625, (2, 2, 4)))


def test_signed_hole():
    vertices = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 1.0],
        ]
    )
    result = _coverage.render(
        vertices, [[0, 1, 2, 3], [4, 5, 6, 7]], [0, 0], np.ones((1, 4)), 3, 3
    )
    expected = np.ones((3, 3, 4))
    expected[1, 1] = 0
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "vertices,loops,colors",
    [
        ([[0.13, 0.27], [2.76, 0.46], [1.48, 2.67]], [[0, 1, 2]], [0]),
        (
            [[-0.31, 0.17], [2.61, -0.23], [3.19, 2.84], [0.42, 3.23]],
            [[0, 1, 2, 3]],
            [0],
        ),
        (
            [
                [0.15, 0.12],
                [2.89, 0.18],
                [2.82, 2.87],
                [0.21, 2.78],
                [0.81, 0.82],
                [0.89, 2.11],
                [2.13, 2.07],
                [2.04, 0.79],
            ],
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [0, 0],
        ),
        (
            [
                [0.13, 0.21],
                [1.41, 0.26],
                [1.54, 2.79],
                [0.21, 2.82],
                [2.87, 0.19],
                [2.81, 2.86],
            ],
            [[0, 1, 2, 3], [1, 4, 5, 2]],
            [0, 1],
        ),
    ],
)
def test_analytic_gradient(vertices, loops, colors):
    vertices = np.array(vertices, dtype=np.float64)
    palette = np.array([[0.2, 0.4, 0.8, 1.0], [0.9, 0.3, 0.1, 0.7]])
    target = np.random.default_rng(51).random((3, 3, 4))
    loss, gradient = _coverage.loss_gradient(
        vertices, loops, colors, palette, 3, 3, target
    )
    rendered = _coverage.render(vertices, loops, colors, palette, 3, 3)
    assert loss == pytest.approx(0.5 * np.sum((rendered - target) ** 2))
    numeric = np.zeros_like(vertices)
    for i in range(len(vertices)):
        for j in range(2):
            plus, minus = vertices.copy(), vertices.copy()
            plus[i, j] += 1e-6
            minus[i, j] -= 1e-6
            lp = _coverage.loss_gradient(plus, loops, colors, palette, 3, 3, target)[0]
            lm = _coverage.loss_gradient(minus, loops, colors, palette, 3, 3, target)[0]
            numeric[i, j] = (lp - lm) / 2e-6
    np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-7)


def test_partition_shared_edge_cancels():
    vertices = np.array(
        [[0.0, 0.0], [1.31, 0.0], [1.63, 3.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]]
    )
    palette = np.ones((1, 4))
    loops = [[0, 1, 2, 3], [1, 4, 5, 2]]
    result = _coverage.render(vertices, loops, [0, 0], palette, 3, 3)
    np.testing.assert_allclose(result, 1.0, atol=1e-14)
    _, gradient = _coverage.loss_gradient(
        vertices, loops, [0, 0], palette, 3, 3, np.zeros((3, 3, 4))
    )
    np.testing.assert_allclose(gradient[[1, 2], 0], 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "mutation", ["nan", "index", "color", "dimension", "palette", "target"]
)
def test_invalid_inputs(mutation):
    vertices = np.array([[0.1, 0.1], [1.9, 0.2], [0.3, 1.8]])
    loops, colors, palette, width = [[0, 1, 2]], [0], np.ones((1, 4)), 2
    target = np.zeros((2, 2, 4))
    if mutation == "nan":
        vertices[0, 0] = np.nan
    if mutation == "index":
        loops = [[0, 1, 8]]
    if mutation == "color":
        colors = [1]
    if mutation == "dimension":
        width = -1
    if mutation == "palette":
        palette = np.ones((1, 3))
    if mutation == "target":
        target[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        _coverage.loss_gradient(vertices, loops, colors, palette, width, 2, target)


def test_large_collinear_loop():
    """Exercise a 2048-node loop without any machine-dependent time assertion."""
    size = 512
    vertices = np.array(
        [(x, 0) for x in range(size)]
        + [(size, y) for y in range(size)]
        + [(x, size) for x in range(size, 0, -1)]
        + [(0, y) for y in range(size, 0, -1)],
        dtype=np.float64,
    )
    loops = [list(range(len(vertices)))]
    palette = np.ones((1, 4))
    rendered = _coverage.render(vertices, loops, [0], palette, size, size)
    np.testing.assert_allclose(rendered, 1.0)
    loss, gradient = _coverage.loss_gradient(
        vertices, loops, [0], palette, size, size, np.zeros((size, size, 4))
    )
    assert loss == pytest.approx(2 * size**2)
    assert np.isfinite(gradient).all()


def test_disconnected_concave_slab():
    # A U has two disconnected intersections with upper horizontal slabs.
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
        ]
    )
    result = _coverage.render(vertices, [list(range(8))], [0], np.ones((1, 4)), 3, 3)
    expected = np.ones((3, 3, 4))
    expected[:2, 1] = 0
    np.testing.assert_allclose(result, expected)
