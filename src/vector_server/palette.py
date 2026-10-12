"""Deterministic palette estimation suppressing blended-edge color samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def nearest_labels(image: FloatArray, palette: FloatArray) -> IntArray:
    flat = image.reshape(-1, 4)
    result = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), 8192):
        block = flat[start : start + 8192]
        result[start : start + len(block)] = np.argmin(
            np.sum((block[:, None, :] - palette[None, :, :]) ** 2, axis=2), axis=1
        )
    return result.reshape(image.shape[:2])


def estimate_palette(
    image: FloatArray, colors: int, detail: float, preset: str
) -> FloatArray:
    flat = image.reshape(-1, 4)
    edge = np.zeros(image.shape[:2], dtype=np.float64)
    edge[1:] += np.linalg.norm(image[1:] - image[:-1], axis=2)
    edge[:, 1:] += np.linalg.norm(image[:, 1:] - image[:, :-1], axis=2)
    weights = np.maximum(1 - 8 * edge, 0.0032).reshape(-1)
    stride = max(1, len(flat) // 16000)
    samples = flat[::stride]
    weights = weights[::stride]
    unique = np.unique(samples, axis=0)
    k = colors or (
        int(4 + detail * 12) if preset == "artwork" else int(6 + detail * 22)
    )
    k = min(k, len(unique))
    # Farthest weighted center initialization, stable ties and ordering.
    centers = [samples[int(np.argmax(weights))].copy()]
    distances = np.sum((samples - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        if not colors and len(centers) >= 2:
            distortion = float(np.average(distances, weights=weights))
            penalty = 0.0003 + (1 - detail) * 0.0015
            if distortion <= penalty:
                break
        idx = int(np.argmax(distances * weights))
        centers.append(samples[idx].copy())
        distances = np.minimum(distances, np.sum((samples - centers[-1]) ** 2, axis=1))
    k = len(centers)
    palette = np.asarray(centers, dtype=np.float64)
    for _ in range(20):
        distances_all = np.sum((samples[:, None, :] - palette[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances_all, axis=1)
        updated = palette.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                updated[j] = np.average(samples[mask], axis=0, weights=weights[mask])
        if np.max(np.abs(updated - palette)) < 1e-5:
            break
        palette = updated
    if not colors and len(palette) > 2:
        # Remove near-redundant centers with an explicit detail-controlled penalty.
        threshold = 0.015 + (1 - detail) * 0.055
        selected = [0]
        for j in range(1, len(palette)):
            if (
                np.min(np.linalg.norm(palette[selected] - palette[j], axis=1))
                > threshold
            ):
                selected.append(j)
        palette = palette[selected]
    return palette


def cleanup_labels(
    labels: IntArray, image: FloatArray, palette: FloatArray, minimum: int
) -> IntArray:
    result = labels.copy()
    for color in np.unique(labels):
        components, count = ndimage.label(labels == color)
        sizes = np.bincount(components.ravel())
        objects = ndimage.find_objects(components)
        for region in np.flatnonzero((sizes < minimum) & (sizes > 0)):
            if region == 0:
                continue
            box = objects[int(region) - 1]
            if box is None:
                continue
            ys, xs = box
            padded = (
                slice(max(0, ys.start - 1), min(labels.shape[0], ys.stop + 1)),
                slice(max(0, xs.start - 1), min(labels.shape[1], xs.stop + 1)),
            )
            mask = components[padded] == region
            border = ndimage.binary_dilation(mask) & ~mask
            candidates = np.unique(result[padded][border])
            candidates = candidates[candidates != color]
            if not len(candidates):
                continue
            mean = np.mean(image[padded][mask], axis=0)
            candidate = int(
                candidates[np.argmin(np.sum((palette[candidates] - mean) ** 2, axis=1))]
            )
            result[padded][mask] = candidate
    return result
