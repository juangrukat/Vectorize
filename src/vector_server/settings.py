"""Environment-controlled server resources."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from vector_server.models import EngineLimits


@dataclass(frozen=True)
class Settings:
    """One API process owns its job volume and child workers."""

    data_dir: Path = field(default_factory=lambda: Path("work/vector-data"))
    workers: int = 2
    max_upload_bytes: int = 16 * 1024 * 1024
    max_pending_jobs: int = 32
    job_timeout_seconds: float = 120
    result_ttl_seconds: float = 24 * 3600
    engine_limits: EngineLimits = field(default_factory=EngineLimits)
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        if min(self.max_upload_bytes, self.max_pending_jobs) < 1:
            raise ValueError("upload and queue limits must be positive")
        if min(self.job_timeout_seconds, self.result_ttl_seconds) <= 0:
            raise ValueError("timeout and retention must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        """Parse explicit deployment settings; fail early on invalid values."""
        return cls(
            data_dir=Path(os.getenv("VECTOR_DATA_DIR", "work/vector-data")),
            workers=int(os.getenv("VECTOR_WORKERS", "2")),
            max_upload_bytes=int(os.getenv("VECTOR_MAX_UPLOAD_BYTES", "16777216")),
            max_pending_jobs=int(os.getenv("VECTOR_MAX_PENDING_JOBS", "32")),
            job_timeout_seconds=float(os.getenv("VECTOR_JOB_TIMEOUT_SECONDS", "120")),
            result_ttl_seconds=float(os.getenv("VECTOR_RESULT_TTL_SECONDS", "86400")),
            engine_limits=EngineLimits(
                max_pixels=int(os.getenv("VECTOR_MAX_PIXELS", "24000000")),
                max_dimension=int(os.getenv("VECTOR_MAX_DIMENSION", "512")),
                max_regions=int(os.getenv("VECTOR_MAX_REGIONS", "2000")),
                max_iterations=int(os.getenv("VECTOR_MAX_ITERATIONS", "30")),
            ),
            api_key=os.getenv("VECTOR_API_KEY") or None,
        )
