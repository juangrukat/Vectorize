"""Local image-to-SVG command with the same engine as HTTP requests."""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast

from vector_server.models import VectorizationError, VectorOptions
from vector_server.settings import Settings


def main(argv: list[str] | None = None) -> int:
    """Convert a local image; report metadata on stderr and errors clearly."""
    from vector_server.engine import vectorize

    parser = argparse.ArgumentParser(description="Convert artwork or photos to SVG")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preset", choices=["artwork", "photo"], default="artwork")
    parser.add_argument("--colors", type=int, default=0, help="0=auto, or 2..64")
    parser.add_argument("--detail", type=float, default=0.7)
    parser.add_argument("--background", default="transparent")
    args = parser.parse_args(argv)
    try:
        config = Settings.from_env()
        data = args.input.read_bytes()
        if len(data) > config.max_upload_bytes:
            raise VectorizationError("image exceeds upload limit")
        options = VectorOptions(
            cast(Literal["artwork", "photo"], args.preset),
            args.colors,
            args.detail,
            args.background,
        )
        result = vectorize(data, options, config.engine_limits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.svg, encoding="utf-8")
    except (VectorizationError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result.metadata, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
