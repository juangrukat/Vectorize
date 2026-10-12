import numpy as np
import pytest

from vector_server.photo_filter import enhance


def test_preserves_step_edges():
    image = np.ones((12, 14, 4))
    image[:, :7, :3] = 0
    assert np.allclose(enhance(image, 3), image)


def test_reduces_small_noise():
    rng = np.random.default_rng(123)
    image = np.ones((48, 48, 4))
    image[..., :3] = np.clip(rng.normal(0.5, 0.025, (48, 48, 3)), 0, 1)
    result = enhance(image, 2)
    assert result[..., :3].std() < image[..., :3].std() * 0.5
    assert abs(result[..., :3].mean() - 0.5) < 0.004


def test_transparent_rgb_cannot_bleed_and_alpha_unchanged():
    image = np.zeros((9, 9, 4))
    image[..., 2] = 1
    image[2:7, 2:7] = [1, 0, 0, 0.5]
    result = enhance(image, 3)
    assert np.array_equal(result[..., 3], image[..., 3])
    assert np.allclose(result[2:7, 2:7, :3], [1, 0, 0])


def test_matches_opaque_selective_weighted_average_at_center():
    rng = np.random.default_rng(39)
    rgb = np.clip(rng.normal(128, 22, (5, 5, 3)), 0, 255)
    image = np.concatenate((rgb / 255, np.ones((5, 5, 1))), axis=2)
    mean = (rgb + rgb[2, 2]) / 2
    delta = (rgb - rgb[2, 2]) ** 2
    distance = (
        (512 + mean[..., 0]) * delta[..., 0] + (767 - mean[..., 2]) * delta[..., 2]
    ) / 255 + 4 * delta[..., 1]
    kernel = np.array(
        [
            [5, 8, 10, 8, 5],
            [8, 20, 40, 20, 8],
            [10, 40, 80, 40, 10],
            [8, 20, 40, 20, 8],
            [5, 8, 10, 8, 5],
        ]
    )
    weights = kernel * (distance < 2601)
    expected = (rgb * weights[..., None]).sum(axis=(0, 1)) / weights.sum() / 255
    assert np.allclose(enhance(image, 1)[2, 2, :3], expected, atol=1e-6)


def test_invalid_input():
    with pytest.raises(ValueError):
        enhance(np.zeros((4, 4, 3)), 1)
    with pytest.raises(ValueError):
        enhance(np.zeros((4, 4, 4)), -1)
