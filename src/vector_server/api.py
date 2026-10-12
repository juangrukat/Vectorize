"""HTTP upload, synchronous conversion and durable job endpoints."""

import asyncio
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from vector_server import __version__
from vector_server.jobs import Job, JobManager, QueueFullError
from vector_server.models import Scalar, VectorizationError, VectorOptions
from vector_server.settings import Settings

logger = logging.getLogger(__name__)


class JobResponse(BaseModel):
    """Public durable job state and links."""

    id: str
    status: Literal["queued", "running", "complete", "failed"]
    created_at: float
    updated_at: float
    metadata: dict[str, Scalar] = Field(default_factory=dict)
    error: str | None = None
    status_url: str
    result_url: str


def public_job(job: Job) -> JobResponse:
    """Serialize state without exposing internal filesystem paths."""
    return JobResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        metadata=job.metadata,
        error=job.error,
        status_url=f"/v1/jobs/{job.id}",
        result_url=f"/v1/jobs/{job.id}/result",
    )


def create_app(
    settings: Settings | None = None, manager: JobManager | None = None
) -> FastAPI:
    """Create one API process; workers and storage are owned by lifespan."""
    config = settings or Settings.from_env()
    jobs = manager or JobManager(config)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        await jobs.start()
        application.state.jobs = jobs
        try:
            yield
        finally:
            await jobs.stop()

    app = FastAPI(
        title="Vectorize",
        version=__version__,
        description="CPU artwork and photo vectorization into editable SVG paths.",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def tester() -> HTMLResponse:
        """Local upload/preview page for testing the conversion API."""
        page = Path(__file__).with_name("tester.html")
        return HTMLResponse(await asyncio.to_thread(page.read_text, encoding="utf-8"))

    def authorize(key: str | None) -> None:
        if config.api_key and (
            key is None or not hmac.compare_digest(key, config.api_key)
        ):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def find(job_id: str) -> Job:
        job = jobs.store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found or expired")
        return job

    async def submit(
        image: UploadFile,
        preset: Literal["artwork", "photo"],
        colors: int,
        detail: float,
        background: str,
        key: str | None,
    ) -> Job:
        authorize(key)
        if not jobs.ready:
            raise HTTPException(503, "conversion workers are not ready")
        try:
            options = VectorOptions(preset, colors, detail, background)
        except VectorizationError as exc:
            raise HTTPException(422, str(exc)) from exc
        chunks: list[bytes] = []
        size = 0
        try:
            while chunk := await image.read(64 * 1024):
                size += len(chunk)
                if size > config.max_upload_bytes:
                    raise HTTPException(413, "image exceeds upload limit")
                chunks.append(chunk)
        finally:
            await image.close()
        if not size:
            raise HTTPException(422, "image is empty")
        try:
            return await asyncio.to_thread(jobs.store.create, b"".join(chunks), options)
        except QueueFullError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "2"}) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Cheap liveness check; does not perform conversion."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        """Scheduler readiness for accepting images."""
        if not jobs.ready:
            raise HTTPException(503, "conversion workers are not ready")
        return {"status": "ready"}

    @app.post("/v1/jobs", status_code=202, response_model=JobResponse)
    async def create_job(
        image: Annotated[UploadFile, File(description="PNG, JPEG or WebP image")],
        preset: Annotated[Literal["artwork", "photo"], Form()] = "artwork",
        colors: Annotated[int, Form()] = 0,
        detail: Annotated[float, Form()] = 0.7,
        background: Annotated[str, Form()] = "transparent",
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> JobResponse:
        """Queue conversion and return stable polling links."""
        return public_job(
            await submit(image, preset, colors, detail, background, x_api_key)
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    async def get_job(
        job_id: str, x_api_key: Annotated[str | None, Header()] = None
    ) -> JobResponse:
        """Read pending, completed or failed conversion state."""
        authorize(x_api_key)
        return public_job(find(job_id))

    @app.get("/v1/jobs/{job_id}/result", response_class=FileResponse)
    async def get_result(
        job_id: str, x_api_key: Annotated[str | None, Header()] = None
    ) -> FileResponse:
        """Download a complete SVG artifact."""
        authorize(x_api_key)
        job = find(job_id)
        if job.status != "complete":
            raise HTTPException(409, f"job is {job.status}")
        path = jobs.store.result_path(job.id)
        if not path.exists():
            raise HTTPException(410, "result is no longer available")
        return FileResponse(
            path,
            media_type="image/svg+xml",
            filename=f"vector-{job.id}.svg",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    async def delete_job(
        job_id: str, x_api_key: Annotated[str | None, Header()] = None
    ) -> Response:
        """Remove a terminal job and its result."""
        authorize(x_api_key)
        job = find(job_id)
        if not jobs.store.delete(job.id):
            raise HTTPException(409, "active jobs cannot be deleted")
        return Response(status_code=204)

    @app.post("/v1/vectorize", response_class=Response)
    async def convert(
        image: Annotated[UploadFile, File(description="PNG, JPEG or WebP image")],
        preset: Annotated[Literal["artwork", "photo"], Form()] = "artwork",
        colors: Annotated[int, Form()] = 0,
        detail: Annotated[float, Form()] = 0.7,
        background: Annotated[str, Form()] = "transparent",
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Use the same bounded worker scheduler and wait for SVG."""
        submitted = await submit(image, preset, colors, detail, background, x_api_key)
        # Queue waiting is separately bounded; running deadlines kill the child.
        deadline = time.monotonic() + config.job_timeout_seconds * (
            config.max_pending_jobs // config.workers + 2
        )
        while time.monotonic() < deadline:
            job = find(submitted.id)
            if job.status == "complete":
                svg = await asyncio.to_thread(jobs.store.result_path(job.id).read_bytes)
                return Response(
                    svg,
                    media_type="image/svg+xml",
                    headers={
                        "Content-Disposition": 'attachment; filename="vector.svg"',
                        "X-Vector-Job-ID": job.id,
                        "X-Vector-Metadata": json.dumps(
                            job.metadata, separators=(",", ":")
                        ),
                        "Cache-Control": "no-store",
                        "X-Content-Type-Options": "nosniff",
                    },
                )
            if job.status == "failed":
                status = 504 if job.error == "conversion exceeded timeout" else 422
                raise HTTPException(status, job.error or "conversion failed")
            await asyncio.sleep(0.05)
        raise HTTPException(504, "queue wait timed out; poll the job endpoint")

    return app


def main() -> None:
    """Run the configured single API process."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, workers=1)
