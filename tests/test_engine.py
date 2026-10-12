import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from vector_server import _coverage
from vector_server.engine import optimize_mesh, vectorize
from vector_server.geometry import extract_mesh
from vector_server.models import EngineLimits, VectorizationError, VectorOptions
from vector_server.palette import cleanup_labels, estimate_palette, nearest_labels


def encode(image, format="PNG"):
    out = io.BytesIO()
    image.save(out, format=format)
    return out.getvalue()


def test_artwork_deterministic_transparency_and_dimensions():
    image = Image.new("RGBA", (32, 24), (255, 0, 255, 0))
    for y in range(5, 19):
        for x in range(7, 25):
            image.putpixel((x, y), (255, 0, 0, 128))
    data = encode(image)
    limits = EngineLimits(max_dimension=16, max_iterations=2)
    result = vectorize(data, VectorOptions(colors=2), limits)
    assert result.svg == vectorize(data, VectorOptions(colors=2), limits).svg
    root = ET.fromstring(result.svg)
    assert root.attrib["viewBox"] == "0 0 32 24"
    assert result.metadata["working_width"] == 16
    assert "image" not in result.svg
    assert "#ff0000" in result.svg
    assert (
        result.metadata["coverage_loss_after"]
        <= result.metadata["coverage_loss_before"]
    )


def test_photo_gradient_palette():
    x = np.linspace(0, 255, 24, dtype=np.uint8)
    pixels = np.empty((24, 24, 3), dtype=np.uint8)
    pixels[:, :, 0] = x
    pixels[:, :, 1] = x[:, None]
    pixels[:, :, 2] = 80
    result = vectorize(
        encode(Image.fromarray(pixels)),
        VectorOptions(preset="photo", colors=4),
        EngineLimits(max_iterations=1, max_dimension=24),
    )
    assert result.metadata["colors"] == 4
    assert result.metadata["curves"] > 0
    assert ET.fromstring(result.svg).tag.endswith("svg")


def test_photo_auto_uses_independent_region_colors():
    values = np.linspace(0, 255, 40, dtype=np.uint8)
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    pixels[:, :, 0] = values
    pixels[:, :, 1] = values[:, None]
    pixels[:, :, 2] = 80
    result = vectorize(
        encode(Image.fromarray(pixels)),
        VectorOptions(preset="photo", colors=0, detail=0.9),
        EngineLimits(max_iterations=2, max_dimension=40),
    )
    assert result.metadata["segmentation_mode"] == "regional_ward"
    assert result.metadata["photo_filter_passes"] == 1
    assert result.metadata["colors"] > 4
    assert (
        result.metadata["coverage_loss_after"]
        <= result.metadata["coverage_loss_before"]
    )
    assert "<path" in result.svg and "<image" not in result.svg


def test_formats_background_and_limits():
    image = Image.new("RGB", (20, 20), "blue")
    for fmt in ("JPEG", "WEBP"):
        assert (
            "<path"
            in vectorize(
                encode(image, fmt),
                VectorOptions(background="white"),
                EngineLimits(max_iterations=0),
            ).svg
        )
    with pytest.raises(VectorizationError, match="pixel limit"):
        vectorize(encode(image), VectorOptions(), EngineLimits(max_pixels=10))
    with pytest.raises(VectorizationError, match="Only PNG"):
        vectorize(encode(image, "BMP"), VectorOptions(), EngineLimits())
    with pytest.raises(VectorizationError, match="Invalid"):
        vectorize(b"invalid", VectorOptions(), EngineLimits())


def test_palette_and_cleanup():
    image = np.zeros((8, 8, 4))
    image[:, :, :3] = 0.5
    image[:, :, 3] = 1
    palette = estimate_palette(image, 0, 0.7, "artwork")
    assert len(palette) == 1
    labels = nearest_labels(image, palette)
    assert np.all(labels == 0)
    labels[3, 3] = 1
    two = np.array([[0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]])
    assert cleanup_labels(labels, image, two, 2)[3, 3] == 0


def test_continuous_reconstruction_improves():
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:, 4:] = 1
    mesh = extract_mesh(labels, 10)
    palette = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    shifted = mesh.vertices.copy()
    interior = (shifted[:, 0] == 4) & ~mesh.pinned
    shifted[interior, 0] += 0.3
    target = _coverage.render(shifted, mesh.loops, mesh.colors, palette, 8, 8)
    result, before, after = optimize_mesh(mesh, palette, target, 15, 1.0)
    assert after < before * 0.1
    np.testing.assert_allclose(result[mesh.pinned], mesh.vertices[mesh.pinned])
    np.testing.assert_array_less(np.abs(result - mesh.vertices), 0.450001)


def test_auto_palette_suppresses_antialias_shades():
    image = np.ones((32, 32, 4), dtype=np.float64)
    image[:, :15, :3] = 0
    image[:, 15, :3] = 0.25
    image[:, 16, :3] = 0.75
    palette = estimate_palette(image, 0, 0.7, "artwork")
    assert len(palette) == 2
    assert np.max(palette[:, 0]) - np.min(palette[:, 0]) > 0.95


def test_fully_transparent_image_and_invalid_profile():
    transparent = Image.new("RGBA", (16, 16), (255, 0, 255, 0))
    result = vectorize(
        encode(transparent), VectorOptions(), EngineLimits(max_iterations=0)
    )
    assert "<path" not in result.svg
    invalid = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(
        invalid, format="PNG", icc_profile=b"invalid"
    )
    with pytest.raises(VectorizationError, match="ICC"):
        vectorize(invalid.getvalue(), VectorOptions(), EngineLimits())


def test_antialiased_circle_compact_cubics():
    from PIL import ImageDraw

    large = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(large).ellipse((24, 24, 232, 232), fill="red")
    image = large.resize((64, 64), Image.Resampling.LANCZOS)
    result = vectorize(
        encode(image),
        VectorOptions(colors=2),
        EngineLimits(max_dimension=64, max_iterations=12),
    )
    assert result.metadata["curves"] < 60


def test_rectangle_has_compact_corner_preserving_paths():
    from PIL import ImageDraw

    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((12, 12, 51, 51), fill="red")
    result = vectorize(
        encode(image),
        VectorOptions(colors=2),
        EngineLimits(max_dimension=64, max_iterations=0),
    )
    # Four outer frame cubics plus a four-cubic shared rectangular interface
    # appearing once for its hole and once for its filled region.
    assert result.metadata["curves"] == 12
