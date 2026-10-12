import numpy as np
import pytest

from vector_server import _coverage
from vector_server.geometry import extract_mesh, signed_area


def test_holes_and_shared_edges():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[1:4, 1:4] = 1
    labels[2, 2] = 0
    mesh = extract_mesh(labels, 10)
    palette = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    rendered = _coverage.render(mesh.vertices, mesh.loops, mesh.colors, palette, 5, 5)
    np.testing.assert_allclose(rendered, palette[labels], atol=1e-10)
    areas = [signed_area(mesh.vertices[loop]) for loop in mesh.loops]
    assert any(a < 0 for a in areas)
    assert sum(areas) == 25
    assert len(mesh.vertices) < sum(map(len, mesh.loops))


def test_diagonal_touching_and_complexity():
    labels = np.indices((4, 4)).sum(axis=0).astype(np.int64) % 2
    with pytest.raises(ValueError, match="connected regions"):
        extract_mesh(labels, 3)
    mesh = extract_mesh(labels, 30)
    assert len(mesh.loops) == 16
    assert np.all(mesh.pinned)


def test_one_pixel():
    mesh = extract_mesh(np.zeros((1, 1), dtype=np.int64), 2)
    assert signed_area(mesh.vertices[mesh.loops[0]]) == 1
    assert np.all(mesh.pinned)
