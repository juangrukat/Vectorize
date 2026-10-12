"""Selective 5x5 denoising translated from recovered preprocessing mathematics.

The original opaque Q8 gate and kernel are retained. Float output avoids byte
rounding after every pass; alpha is preserved and gates use premultiplied color
so transparent RGB cannot bleed into visible pixels.
"""

import numpy as np
from numpy.typing import NDArray

_KERNEL = np.array(
    [
        [5, 8, 10, 8, 5],
        [8, 20, 40, 20, 8],
        [10, 40, 80, 40, 10],
        [8, 20, 40, 20, 8],
        [5, 8, 10, 8, 5],
    ],
    dtype=np.float32,
)


def enhance(image: NDArray[np.floating], passes: int) -> NDArray[np.float64]:
    """Denoise straight RGBA in [0,1], preserving alpha and strong boundaries."""
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("enhance expects an HxWx4 image")
    if passes < 0 or passes > 8:
        raise ValueError("passes must be between 0 and 8")
    if not np.isfinite(image).all():
        raise ValueError("image must contain finite values")
    result = np.clip(image, 0.0, 1.0).astype(np.float32)
    alpha = result[..., 3:4].copy()
    for _ in range(passes):
        premul = result[..., :3] * alpha
        gate = np.concatenate((255.0 * premul, 255.0 * (1.0 - alpha)), axis=2)
        padded_gate = np.pad(gate, ((2, 2), (2, 2), (0, 0)), mode="edge")
        padded_color = np.pad(premul, ((2, 2), (2, 2), (0, 0)), mode="edge")
        padded_alpha = np.pad(alpha, ((2, 2), (2, 2), (0, 0)), mode="edge")
        h, w = result.shape[:2]
        total = np.zeros_like(premul)
        weights = np.zeros_like(alpha)
        for dy in range(5):
            for dx in range(5):
                neighbor = padded_gate[dy : dy + h, dx : dx + w]
                delta = (neighbor - gate) ** 2
                mean = (neighbor + gate) * 0.5
                distance = (
                    (512.0 + mean[..., 0]) * delta[..., 0]
                    + (767.0 - mean[..., 2]) * delta[..., 2]
                    + (767.0 - mean[..., 3]) * delta[..., 3]
                ) / 255.0 + 4.0 * delta[..., 1]
                neighbor_alpha = padded_alpha[dy : dy + h, dx : dx + w]
                admitted = (
                    (distance < 2601.0)
                    & (np.abs(neighbor_alpha[..., 0] - alpha[..., 0]) <= 0.05)
                    & (neighbor_alpha[..., 0] > 1e-6)
                )
                weight = _KERNEL[dy, dx] * admitted[..., None]
                total += padded_color[dy : dy + h, dx : dx + w] * weight
                weights += neighbor_alpha * weight
        np.divide(total, weights, out=result[..., :3], where=weights > 1e-6)
        result[..., 3:4] = alpha
    return np.asarray(result, dtype=np.float64)
