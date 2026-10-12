"""Independent checks of Ward merge costs, topology and mean reconstruction."""

import numpy as np
import pytest
from scipy.ndimage import label

from vector_server import _regions


def test_exact_two_pixel_ward_cost():
    image = np.array([[[0.2, 0.3, 0.4, 1.0], [0.6, 0.3, 0.4, 1.0]]])
    # nA*nB/(nA+nB) * ||muA-muB||^2 = .5 * .4^2 = .08.
    labels, palette = _regions.ward(image, 0.0799, 1, 20)
    assert len(palette) == 2
    np.testing.assert_array_equal(labels, [[0, 1]])
    labels, palette = _regions.ward(image, 0.0801, 1, 20)
    np.testing.assert_array_equal(labels, [[0, 0]])
    np.testing.assert_allclose(palette, [[0.4, 0.3, 0.4, 1.0]])


def test_partition_region_means_and_connectivity():
    image = np.random.default_rng(42).random((24, 32, 4))
    labels, palette = _regions.ward(image, 0.1, 4, 50)
    assert labels.dtype == np.int64
    assert labels.shape == image.shape[:2]
    assert 1 <= len(palette) <= 50
    np.testing.assert_array_equal(np.unique(labels), np.arange(len(palette)))
    for i, color in enumerate(palette):
        pixels = labels == i
        assert label(pixels)[1] == 1
        np.testing.assert_allclose(color, image[pixels].mean(axis=0), atol=1e-14)
        assert pixels.sum() >= 4


def test_independent_continuous_colors():
    # Adjacent gradients retain independent, non-quantized means.
    image = np.ones((1, 20, 4))
    image[0, :, 0] = np.linspace(0.0, 1.0, 20)
    labels, palette = _regions.ward(image, 0.0001, 1, 100)
    assert len(palette) == 20
    np.testing.assert_allclose(palette[:, 0], image[0, :, 0])
    assert len(np.unique(labels)) == 20


def test_noise_cleanup():
    image = np.zeros((20, 20, 4))
    image[..., 3] = 1.0
    image[10, 10, :3] = 1.0
    labels, palette = _regions.ward(image, 0.01, 4, 100)
    assert len(palette) == 1
    np.testing.assert_array_equal(labels, 0)
    np.testing.assert_allclose(palette, [[1 / 400, 1 / 400, 1 / 400, 1]])


def test_deterministic_agglomeration():
    image = np.random.default_rng(2).random((15, 17, 4))
    first = _regions.ward(image, 0.13, 3, 45)
    second = _regions.ward(image, 0.13, 3, 45)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_region_cap():
    image = np.random.default_rng(5).random((12, 12, 4))
    labels, palette = _regions.ward(image, 0.0, 1, 7)
    assert len(palette) == 7
    assert np.unique(labels).size == 7


@pytest.mark.parametrize(
    "invalid",
    ["shape", "nan", "range", "threshold", "minimum", "maximum"],
)
def test_invalid_inputs(invalid):
    image = np.ones((2, 2, 4))
    threshold, minimum, maximum = 0.1, 1, 10
    if invalid == "shape":
        image = image[..., :3]
    elif invalid == "nan":
        image[0, 0, 0] = np.nan
    elif invalid == "range":
        image[0, 0, 0] = 1.1
    elif invalid == "threshold":
        threshold = -0.1
    elif invalid == "minimum":
        minimum = 0
    elif invalid == "maximum":
        maximum = 0
    with pytest.raises(ValueError):
        _regions.ward(image, threshold, minimum, maximum)


def test_boundary_refinement_reduces_sse_and_preserves_connectivity():
    image = np.random.default_rng(0).random((16, 16, 4))
    before, palette_before = _regions.ward(image, 0.2, 2, 30, refine_passes=0)
    after, palette_after = _regions.ward(image, 0.2, 2, 30, refine_passes=2)
    initial_sse = np.sum((image - palette_before[before]) ** 2)
    final_sse = np.sum((image - palette_after[after]) ** 2)
    assert np.any(before != after)
    assert final_sse < initial_sse
    assert len(palette_before) == len(palette_after)
    for region, color in enumerate(palette_after):
        pixels = after == region
        assert label(pixels)[1] == 1
        np.testing.assert_allclose(color, image[pixels].mean(axis=0), atol=1e-14)


@pytest.mark.parametrize("passes", [-1, 5])
def test_refinement_pass_bound(passes):
    with pytest.raises(ValueError):
        _regions.ward(np.ones((2, 2, 4)), 0.1, 1, 10, refine_passes=passes)
