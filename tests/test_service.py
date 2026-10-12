"""Service contracts, durable storage and real worker behavior."""

import asyncio
import io
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vector_server.api import create_app
from vector_server.cli import main as cli_main
from vector_server.jobs import JobManager, JobStore, QueueFullError
from vector_server.models import EngineLimits, VectorizationError, VectorOptions
from vector_server.settings import Settings


def png() -> bytes:
    image = Image.new("RGBA", (16, 16), "white")
    for y in range(4, 12):
        for x in range(4, 12):
            image.putpixel((x, y), (20, 70, 170, 255))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        workers=1,
        job_timeout_seconds=15,
        engine_limits=EngineLimits(max_dimension=32, max_iterations=1),
    )


def wait(
    client: TestClient, job_id: str, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        reply = client.get(f"/v1/jobs/{job_id}", headers=headers)
        body = reply.json()
        if body["status"] in {"complete", "failed"}:
            return body  # type: ignore[no-any-return]
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_store_capacity_recovery_and_expiration(tmp_path: Path) -> None:
    store = JobStore(tmp_path, 1)
    job = store.create(png(), VectorOptions())
    with pytest.raises(QueueFullError):
        store.create(png(), VectorOptions())
    assert len(list(tmp_path.glob("*.input"))) == 1
    store.update(job.id, "running")
    restarted = JobStore(tmp_path, 1)
    restarted.recover()
    assert restarted.get(job.id).status == "queued"  # type: ignore[union-attr]
    store.input_path(job.id).unlink()
    restarted.recover()
    assert restarted.get(job.id).status == "failed"  # type: ignore[union-attr]
    assert restarted.cleanup(-1) == 1
    assert restarted.get(job.id) is None
    assert not restarted.delete(job.id)


def test_store_terminal_result_and_active_deletion(tmp_path: Path) -> None:
    store = JobStore(tmp_path, 2)
    job = store.create(png(), VectorOptions("photo", 3, 0.4, "white"))
    assert not store.delete(job.id)
    store.result_path(job.id).write_text("<svg/>")
    store.update(job.id, "complete", {"paths": 3})
    assert store.get(job.id).metadata == {"paths": 3}  # type: ignore[union-attr]
    assert not store.input_path(job.id).exists()
    assert store.delete(job.id)
    assert not store.result_path(job.id).exists()
    with pytest.raises(ValueError):
        store.input_path("../../bad")


def test_real_job_and_persisted_result(config: Settings) -> None:
    with TestClient(create_app(config)) as client:
        assert "Image to vector" in client.get("/").text
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        response = client.post("/v1/jobs", files={"image": ("input.png", png())})
        assert response.status_code == 202
        job_id = response.json()["id"]
        pending = client.get(f"/v1/jobs/{job_id}/result")
        assert pending.status_code in {200, 409}
        done = wait(client, job_id)
        assert done["status"] == "complete", done
        output = client.get(done["result_url"])
        assert output.status_code == 200
        assert "<svg" in output.text and "<path" in output.text
        assert "<image" not in output.text
    with TestClient(create_app(config)) as client:
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "complete"
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_sync_photo_and_invalid_image(config: Settings) -> None:
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/vectorize",
            files={"image": ("x.png", png())},
            data={"preset": "photo", "colors": 3, "detail": 0.3},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("image/svg+xml")
        assert json.loads(response.headers["x-vector-metadata"])
        invalid = client.post("/v1/vectorize", files={"image": ("x.png", b"garbage")})
        assert invalid.status_code == 422


def test_auth_limits_and_invalid_options(config: Settings) -> None:
    config = replace(config, api_key="test-key", max_upload_bytes=200)
    headers = {"X-API-Key": "test-key"}
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/jobs/missing").status_code == 401
        assert client.get("/v1/jobs/missing", headers=headers).status_code == 404
        for data in [{"colors": 1}, {"detail": 2}, {"background": "invalid"}]:
            reply = client.post(
                "/v1/jobs", files={"image": ("x", png())}, data=data, headers=headers
            )
            assert reply.status_code == 422
        assert (
            client.post(
                "/v1/jobs", files={"image": ("x", b"")}, headers=headers
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/jobs", files={"image": ("x", b"a" * 201)}, headers=headers
            ).status_code
            == 413
        )


def test_api_missing_result_and_active_delete(config: Settings) -> None:
    manager = JobManager(config)
    with TestClient(create_app(config, manager)) as client:
        job = manager.store.create(png(), VectorOptions())
        assert client.delete(f"/v1/jobs/{job.id}").status_code == 409
        manager.store.update(job.id, "complete")
        assert client.get(f"/v1/jobs/{job.id}/result").status_code == 410


def test_settings_validation_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_WORKERS", "3")
    monkeypatch.setenv("VECTOR_MAX_ITERATIONS", "2")
    assert Settings.from_env().workers == 3
    assert Settings.from_env().engine_limits.max_iterations == 2
    for kwargs in [{"workers": 0}, {"max_pending_jobs": 0}, {"job_timeout_seconds": 0}]:
        with pytest.raises(ValueError):
            Settings(**kwargs)  # type: ignore[arg-type]
    for kwargs in [{"max_dimension": 1}, {"max_regions": 0}]:
        with pytest.raises(ValueError):
            EngineLimits(**kwargs)
    with pytest.raises(VectorizationError):
        VectorOptions("invalid")  # type: ignore[arg-type]


def test_exclusive_volume_owner(config: Settings) -> None:
    async def exercise() -> None:
        first = JobManager(config)
        second = JobManager(config)
        await first.start()
        try:
            with pytest.raises(RuntimeError, match="already owns"):
                await second.start()
        finally:
            await first.stop()

    asyncio.run(exercise())


def slow_worker(*args: Any) -> None:
    time.sleep(5)


def crashed_worker(*args: Any) -> None:
    os._exit(7)


def test_timeout_kills_child_and_returns_504(config: Settings) -> None:
    config = replace(config, job_timeout_seconds=0.3)
    manager = JobManager(config, slow_worker)
    with TestClient(create_app(config, manager)) as client:
        reply = client.post("/v1/vectorize", files={"image": ("x.png", png())})
        assert reply.status_code == 504
        assert "timeout" in reply.text
        deadline = time.monotonic() + 3
        while manager.active and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not manager.active


def test_crashed_worker_and_not_ready(config: Settings) -> None:
    manager = JobManager(config, crashed_worker)
    with TestClient(create_app(config, manager)) as client:
        reply = client.post("/v1/jobs", files={"image": ("x.png", png())})
        assert wait(client, reply.json()["id"])["error"] == "conversion worker exited"
        manager.ready = False
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/jobs", files={"image": ("x", png())}).status_code == 503


def test_shutdown_requeues_inflight_work(config: Settings) -> None:
    manager = JobManager(config, slow_worker)
    with TestClient(create_app(config, manager)) as client:
        job_id = client.post("/v1/jobs", files={"image": ("x.png", png())}).json()["id"]
        deadline = time.monotonic() + 5
        while client.get(f"/v1/jobs/{job_id}").json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
    assert manager.store.get(job_id).status == "queued"  # type: ignore[union-attr]
    assert not manager.active


def test_cli_real_conversion_and_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(png())
    destination = tmp_path / "sub" / "out.svg"
    monkeypatch.setenv("VECTOR_MAX_ITERATIONS", "0")
    assert cli_main([str(source), str(destination), "--colors", "2"]) == 0
    assert "<path" in destination.read_text()
    assert json.loads(capsys.readouterr().err)["colors"] == 2
    assert cli_main([str(tmp_path / "missing"), str(destination)]) == 2
    monkeypatch.setenv("VECTOR_MAX_UPLOAD_BYTES", "1")
    assert cli_main([str(source), str(destination)]) == 2
    assert "upload limit" in capsys.readouterr().err


def test_store_caps_retained_history(tmp_path: Path) -> None:
    store = JobStore(tmp_path, 1)
    oldest = ""
    for index in range(257):
        job = store.create(b"input", VectorOptions())
        if index == 0:
            oldest = job.id
        store.update(job.id, "failed", error="bad image")
    assert store.cleanup(86400) == 1
    assert store.get(oldest) is None
