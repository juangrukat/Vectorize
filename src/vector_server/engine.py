"""Independent raster-to-vector pipeline using exact pixel-area reconstruction.

Colors and loss use premultiplied RGBA in sRGB numeric space, not linear light.
This explicit convention preserves transparent edges without hidden RGB halos.
The bounded optimization combines coverage loss with a tangent turning prior.
"""

from __future__ import annotations

import io
import time
import warnings

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy.optimize import minimize

from . import _coverage, _regions
from .curves import mesh_paths
from .geometry import Mesh, extract_mesh
from .models import EngineLimits, VectorizationError, VectorOptions, VectorResult
from .palette import cleanup_labels, estimate_palette, nearest_labels
from .photo_filter import enhance

FloatArray = NDArray[np.float64]


def _decode(
    data: bytes, options: VectorOptions, limits: EngineLimits
) -> tuple[FloatArray, int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise VectorizationError("Only PNG, JPEG and WebP images are supported")
            if image.width * image.height > limits.max_pixels:
                raise VectorizationError("Decoded image exceeds pixel limit")
            image = ImageOps.exif_transpose(image)
            profile = image.info.get("icc_profile")
            if profile:
                try:
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGBA",
                    )
                except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                    raise VectorizationError(
                        "Unsupported or invalid ICC color profile"
                    ) from exc
            image = image.convert("RGBA")
            width, height = image.size
            if options.background != "transparent":
                background = Image.new("RGBA", image.size, options.background)
                background.alpha_composite(image)
                image = background
            dimension = min(
                limits.max_dimension,
                getattr(options, "max_dimension", limits.max_dimension),
            )
            image.thumbnail((dimension, dimension), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float64) / 255
            pixels[:, :, :3] *= pixels[:, :, 3:4]
            return pixels, width, height
    except VectorizationError:
        raise
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise VectorizationError("Invalid or unsafe image") from exc


def optimize_mesh(
    mesh: Mesh, palette: FloatArray, target: FloatArray, iterations: int, detail: float
) -> tuple[FloatArray, float, float]:
    base = mesh.vertices.copy()
    h, w = target.shape[:2]
    before, _ = _coverage.loss_gradient(
        base, mesh.loops, mesh.colors, palette, w, h, target
    )
    free = np.flatnonzero(~mesh.pinned)
    if not len(free) or iterations <= 0:
        return base, float(before), float(before)
    # Curvature regularizer uses sqrt(2 - 2 cos(theta) + epsilon).
    triples = np.asarray(
        [
            (loop[i - 1], v, loop[(i + 1) % len(loop)])
            for loop in mesh.loops
            for i, v in enumerate(loop)
        ],
        dtype=np.int64,
    )
    weight = 0.003 * (1 - detail)

    def objective(x: FloatArray) -> tuple[float, FloatArray]:
        vertices = base.copy()
        vertices[free] = x.reshape(-1, 2)
        loss, gradient = _coverage.loss_gradient(
            vertices, mesh.loops, mesh.colors, palette, w, h, target
        )
        grad = np.asarray(gradient, dtype=np.float64)
        a = vertices[triples[:, 1]] - vertices[triples[:, 0]]
        b = vertices[triples[:, 2]] - vertices[triples[:, 1]]
        na = np.maximum(np.linalg.norm(a, axis=1), 1e-9)
        nb = np.maximum(np.linalg.norm(b, axis=1), 1e-9)
        cosine = np.clip(np.sum(a * b, axis=1) / (na * nb), -1, 1)
        prior = np.sqrt(2 - 2 * cosine + 0.001)
        da = (
            -(b / (na * nb)[:, None] - cosine[:, None] * a / (na * na)[:, None])
            / prior[:, None]
            * weight
        )
        db = (
            -(a / (na * nb)[:, None] - cosine[:, None] * b / (nb * nb)[:, None])
            / prior[:, None]
            * weight
        )
        np.add.at(grad, triples[:, 0], -da)
        np.add.at(grad, triples[:, 1], da - db)
        np.add.at(grad, triples[:, 2], db)
        return float(loss + weight * np.sum(prior)), grad[free].reshape(-1)

    x0 = base[free].reshape(-1)
    fit = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(x0 - 0.45, x0 + 0.45, strict=True)),
        options={"maxiter": iterations, "ftol": 1e-7, "maxls": 12},
    )
    result = base.copy()
    result[free] = fit.x.reshape(-1, 2)
    after, _ = _coverage.loss_gradient(
        result, mesh.loops, mesh.colors, palette, w, h, target
    )
    if after > before:
        return base, float(before), float(before)
    return result, float(before), float(after)


def vectorize(
    data: bytes, options: VectorOptions, limits: EngineLimits
) -> VectorResult:
    started = time.monotonic()
    image, width, height = _decode(data, options, limits)
    regional_photo = options.preset == "photo" and options.colors == 0
    photo_passes = 0
    segmentation_lambda = 0.0
    if regional_photo:
        # Unrestricted photo regions have independently estimated colors.
        # Construct the palette after segmentation, rather than imposing one.
        photo_passes = int(np.clip(round(3 - 2 * options.detail), 1, 3))
        straight = image.copy()
        np.divide(
            image[:, :, :3],
            image[:, :, 3:4],
            out=straight[:, :, :3],
            where=image[:, :, 3:4] > 1e-8,
        )
        filtered = enhance(straight, photo_passes)
        filtered[:, :, :3] *= filtered[:, :, 3:4]
        segmentation_lambda = 10 ** (1 - 2 * options.detail)
        minimum = int(round(2 + 5 * options.detail))
        labels, palette = _regions.ward(
            filtered, segmentation_lambda, minimum, limits.max_regions
        )
    else:
        palette = estimate_palette(
            image, options.colors, options.detail, options.preset
        )
        labels = nearest_labels(image, palette)
        minimum = max(1, int(1 + (1 - options.detail) * 8))
        if options.preset == "photo":
            minimum = max(minimum, int(4 + (1 - options.detail) * 16))
        labels = cleanup_labels(labels, image, palette, minimum)
    while True:
        try:
            mesh = extract_mesh(labels, limits.max_regions)
            break
        except ValueError as exc:
            if regional_photo or options.preset != "photo" or minimum >= 128:
                raise VectorizationError(str(exc)) from exc
            minimum *= 2
            labels = cleanup_labels(labels, image, palette, minimum)
    if len(mesh.vertices) > 60000:
        raise VectorizationError(
            "Image boundary complexity exceeds limit; "
            "reduce working dimensions or detail"
        )
    vertices, before, after = optimize_mesh(
        mesh, palette, image, limits.max_iterations, options.detail
    )
    h, w = image.shape[:2]
    scale = np.array([width / w, height / h])
    paths = []
    curve_count = 0
    tolerance = 0.2 + (1 - options.detail) * 0.8
    fitted_paths = mesh_paths(vertices, mesh.loops, mesh.pinned, tolerance)
    for color in sorted(set(mesh.colors)):
        paths_for_color = [
            path
            for path, c in zip(fitted_paths, mesh.colors, strict=True)
            if c == color
        ]
        path = " ".join(paths_for_color)
        rgba = palette[color]
        alpha = float(rgba[3])
        if alpha < 1e-6:
            continue
        rgb = np.clip(np.rint(rgba[:3] / alpha * 255), 0, 255).astype(int)
        fill = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        paths.append(
            f'<path fill="{fill}" fill-opacity="{alpha:.6f}" '
            f'fill-rule="nonzero" d="{path}"/>'
        )
        curve_count += path.count(" C")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<g transform="scale({scale[0]:.9f} {scale[1]:.9f})">'
        + "".join(paths)
        + "</g></svg>"
    )
    return VectorResult(
        svg=svg,
        metadata={
            "width": width,
            "height": height,
            "working_width": w,
            "working_height": h,
            "colors": len(palette),
            "minimum_region_pixels": minimum,
            "curve_tolerance_pixels": tolerance,
            "loops": len(mesh.loops),
            "curves": curve_count,
            "vertices": len(vertices),
            "coverage_loss_before": before,
            "coverage_loss_after": after,
            "seconds": time.monotonic() - started,
            "preset": options.preset,
            "segmentation_mode": (
                "regional_ward" if regional_photo else "limited_palette"
            ),
            "photo_filter_passes": photo_passes,
            "segmentation_lambda": segmentation_lambda,
        },
    )
