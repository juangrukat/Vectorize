"""Durable bounded jobs, with one killable child process per conversion."""

import asyncio
import fcntl
import json
import logging
import multiprocessing as mp
import queue
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from typing import IO, Literal, cast

from vector_server.models import EngineLimits, Scalar, VectorizationError, VectorOptions
from vector_server.settings import Settings

logger = logging.getLogger(__name__)
Status = Literal["queued", "running", "complete", "failed"]


class QueueFullError(RuntimeError):
    """The bounded pending queue has no capacity."""


@dataclass(frozen=True)
class Job:
    """Persistent public job state."""

    id: str
    status: Status
    created_at: float
    updated_at: float
    options: VectorOptions
    metadata: dict[str, Scalar]
    error: str | None


class JobStore:
    """SQLite transactions and atomic input/result files on one local volume."""

    def __init__(self, root: Path, capacity: int) -> None:
        self.root = root
        self.capacity = capacity
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite3"
        with closing(self._connect()) as con, con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, "
                "status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, "
                "options TEXT NOT NULL, metadata TEXT NOT NULL, error TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def input_path(self, job_id: str) -> Path:
        """Return a generated-ID path, never a caller-controlled filename."""
        return self.root / f"{uuid.UUID(job_id).hex}.input"

    def result_path(self, job_id: str) -> Path:
        """Return the atomic SVG destination."""
        return self.root / f"{uuid.UUID(job_id).hex}.svg"

    def create(self, data: bytes, options: VectorOptions) -> Job:
        """Reserve queue capacity transactionally, then commit the input."""
        job_id = uuid.uuid4().hex
        now = time.time()
        path = self.input_path(job_id)
        try:
            with closing(self._connect()) as con, con:
                con.execute("BEGIN IMMEDIATE")
                pending = int(
                    con.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
                    ).fetchone()[0]
                )
                if pending >= self.capacity:
                    raise QueueFullError("conversion queue is full")
                path.write_bytes(data)
                con.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                    (
                        job_id,
                        "queued",
                        now,
                        now,
                        json.dumps(asdict(options)),
                        "{}",
                        None,
                    ),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return Job(job_id, "queued", now, now, options, {}, None)

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        raw = cast(dict[str, object], json.loads(str(row["options"])))
        options = VectorOptions(
            preset=cast(Literal["artwork", "photo"], raw["preset"]),
            colors=int(cast(int, raw["colors"])),
            detail=float(cast(float, raw["detail"])),
            background=str(raw["background"]),
        )
        return Job(
            str(row["id"]),
            cast(Status, row["status"]),
            float(row["created"]),
            float(row["updated"]),
            options,
            cast(dict[str, Scalar], json.loads(str(row["metadata"]))),
            cast(str | None, row["error"]),
        )

    def get(self, job_id: str) -> Job | None:
        """Read a job without interpreting its ID as a filesystem path."""
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def queued(self, limit: int) -> list[Job]:
        """Return pending requests oldest first."""
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._job(row) for row in rows]

    def update(
        self,
        job_id: str,
        status: Status,
        metadata: dict[str, Scalar] | None = None,
        error: str | None = None,
    ) -> None:
        """Record a state transition and release completed input data."""
        with closing(self._connect()) as con, con:
            con.execute(
                "UPDATE jobs SET status=?,updated=?,metadata=?,error=? WHERE id=?",
                (status, time.time(), json.dumps(metadata or {}), error, job_id),
            )
        if status in {"complete", "failed"}:
            self.input_path(job_id).unlink(missing_ok=True)

    def recover(self) -> None:
        """Retry interrupted requests on restart; fail missing inputs explicitly."""
        with closing(self._connect()) as con, con:
            con.execute("UPDATE jobs SET status='queued' WHERE status='running'")
        for job in self.queued(self.capacity):
            if not self.input_path(job.id).exists():
                self.update(job.id, "failed", error="input was lost before conversion")

    def delete(self, job_id: str) -> bool:
        """Delete terminal jobs and their artifacts; active jobs cannot be deleted."""
        with closing(self._connect()) as con, con:
            changed = con.execute(
                "DELETE FROM jobs WHERE id=? AND status IN ('complete','failed')",
                (job_id,),
            ).rowcount
        if changed:
            self.input_path(job_id).unlink(missing_ok=True)
            self.result_path(job_id).unlink(missing_ok=True)
        return bool(changed)

    def cleanup(self, ttl: float) -> int:
        """Expire terminal artifacts and cap retained history at 256 requests."""
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT id,updated FROM jobs WHERE status IN ('complete','failed') "
                "ORDER BY updated DESC"
            ).fetchall()
        expired = [
            str(row["id"])
            for index, row in enumerate(rows)
            if float(row["updated"]) < time.time() - ttl or index >= 256
        ]
        for job_id in expired:
            self.delete(job_id)
        return len(expired)


def run_conversion(
    root: str,
    job_id: str,
    options: VectorOptions,
    limits: EngineLimits,
    messages: "Queue[dict[str, object]]",
) -> None:
    """Child entry point; no engine work runs in the HTTP event loop."""
    from vector_server.engine import vectorize

    path = Path(root)
    result = path / f"{job_id}.svg"
    temporary = path / f"{job_id}.svg.tmp"
    try:
        converted = vectorize((path / f"{job_id}.input").read_bytes(), options, limits)
        temporary.write_text(converted.svg, encoding="utf-8")
        temporary.replace(result)
        messages.put({"ok": True, "metadata": converted.metadata})
    except VectorizationError as exc:
        messages.put({"ok": False, "error": str(exc)})
    except Exception:
        logger.exception("conversion failed for job %s", job_id)
        messages.put({"ok": False, "error": "conversion failed unexpectedly"})
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ActiveJob:
    """Child process ownership and monotonic deadline."""

    process: BaseProcess
    messages: "Queue[dict[str, object]]"
    started: float


class JobManager:
    """Bounded process scheduler with enforceable per-job deadlines."""

    def __init__(
        self,
        settings: Settings,
        worker: Callable[
            [str, str, VectorOptions, EngineLimits, "Queue[dict[str, object]]"], None
        ] = run_conversion,
    ) -> None:
        self.settings = settings
        self.store = JobStore(settings.data_dir, settings.max_pending_jobs)
        self.worker = worker
        self.context = mp.get_context("spawn")
        self.active: dict[str, ActiveJob] = {}
        self.task: asyncio.Task[None] | None = None
        self.lock: IO[bytes] | None = None
        self.ready = False

    async def start(self) -> None:
        """Take exclusive volume ownership and recover the durable queue."""
        self.lock = (self.settings.data_dir / "owner.lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            self.lock = None
            raise RuntimeError(
                "another API process already owns this job volume"
            ) from exc
        self.store.recover()
        self.store.cleanup(self.settings.result_ttl_seconds)
        self.ready = True
        self.task = asyncio.create_task(self._loop())

    @staticmethod
    def _stop(child: ActiveJob) -> None:
        if child.process.is_alive():
            child.process.terminate()
            child.process.join(timeout=1)
            if child.process.is_alive():
                child.process.kill()
        child.process.join(timeout=1)
        child.messages.close()
        child.messages.cancel_join_thread()
        child.process.close()

    async def stop(self) -> None:
        """Requeue active requests and stop all owned children on shutdown."""
        self.ready = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        for job_id, child in self.active.items():
            self._stop(child)
            self.store.update(job_id, "queued")
        self.active.clear()
        if self.lock:
            self.lock.close()
            self.lock = None

    async def _loop(self) -> None:
        last_cleanup = time.monotonic()
        try:
            while True:
                self.tick()
                if time.monotonic() - last_cleanup > 60:
                    self.store.cleanup(self.settings.result_ttl_seconds)
                    last_cleanup = time.monotonic()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.ready = False
            logger.exception("job scheduler stopped unexpectedly")

    def tick(self) -> None:
        """Collect completions, enforce deadlines and launch queued work."""
        for job_id, child in list(self.active.items()):
            try:
                message = child.messages.get_nowait()
            except queue.Empty:
                message = None
            if message:
                if message.get("ok") is True:
                    self.store.update(
                        job_id,
                        "complete",
                        cast(dict[str, Scalar], message.get("metadata", {})),
                    )
                else:
                    self.store.update(job_id, "failed", error=str(message["error"]))
            elif time.monotonic() - child.started > self.settings.job_timeout_seconds:
                self.store.update(job_id, "failed", error="conversion exceeded timeout")
            elif not child.process.is_alive():
                self.store.update(job_id, "failed", error="conversion worker exited")
            else:
                continue
            self._stop(child)
            del self.active[job_id]
        available = self.settings.workers - len(self.active)
        for job in self.store.queued(available):
            messages: Queue[dict[str, object]] = self.context.Queue()
            process = self.context.Process(
                target=self.worker,
                args=(
                    str(self.settings.data_dir),
                    job.id,
                    job.options,
                    self.settings.engine_limits,
                    messages,
                ),
                daemon=True,
            )
            self.store.update(job.id, "running")
            try:
                process.start()
            except Exception:
                messages.close()
                self.store.update(job.id, "failed", error="could not launch worker")
                logger.exception("worker launch failed")
            else:
                self.active[job.id] = ActiveJob(process, messages, time.monotonic())
