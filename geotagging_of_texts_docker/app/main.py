from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import traceback
import uuid
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import CHAT_MODEL, OCR_MODEL, RERANKER_MODEL, device, gpu_available
from .llm import llama
from .pipeline import (
    pipeline_model_status,
    preload_pipeline_models,
    release_pipeline_models,
    run_pipeline,
)
from .gazetteer import load_gazetteer, lookup_place_by_id
from .tei import write_notebook_tei

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
# SAVE_RUNS=false keeps runs only in the container's temporary folder (RAM-backed, gone on stop).
SAVE_RUNS = os.getenv("SAVE_RUNS", "true").strip().lower() not in ("false", "0", "no", "off")
JOBS_DIR = (DATA_DIR if SAVE_RUNS else Path(tempfile.gettempdir())) / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UNSAVED_RUN_TTL_SECONDS = 3600
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
CLAMSCAN_BIN = os.getenv("CLAMSCAN_BIN", "clamscan")
CLAMAV_DB_DIR = Path(os.getenv("CLAMAV_DB_DIR", "/var/lib/clamav"))
MAX_ACTIVE_JOBS = 8  # queued + running; stops an upload flood from piling up in memory and on disk

# Leaflet and fonts are the only third-party resources the page loads.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data: https://unpkg.com https://*.tile.openstreetmap.org; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    # OpenStreetMap's tile servers refuse requests that carry no Referer at all.
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"Starting on {device().upper()}. Preloading every model before accepting requests.", flush=True)
        await asyncio.to_thread(preload_pipeline_models, lambda message: print(message, flush=True))
        await asyncio.to_thread(llama.start_all, CHAT_MODEL, RERANKER_MODEL, OCR_MODEL)
        print("\n==============================================\n"
              f" The models are ready. Go to http://localhost:{os.getenv('APP_PORT', '8000')}\n"
              "==============================================\n", flush=True)
        yield
    finally:
        await asyncio.to_thread(llama.close)
        release_pipeline_models()


# The interactive API docs are not needed by the web UI.
app = FastAPI(title="Geotagging of Texts", version="1.0.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.RLock()
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _checked_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_id


def _purge_unsaved_runs() -> None:
    """With SAVE_RUNS off, delete runs that are older than an hour so they never pile up."""
    if SAVE_RUNS:
        return
    cutoff = time.time() - UNSAVED_RUN_TTL_SECONDS
    for run in JOBS_DIR.iterdir():
        if run.is_dir() and run.stat().st_mtime < cutoff:
            shutil.rmtree(run, ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(run.name, None)


def _scan_upload(content: bytes) -> None:
    """Reject infected uploads and fail closed if ClamAV is unavailable."""
    with tempfile.NamedTemporaryFile(prefix="geotagging-upload-") as sample:
        sample.write(content)
        sample.flush()
        try:
            result = subprocess.run(
                [CLAMSCAN_BIN, "--no-summary", "--database", str(CLAMAV_DB_DIR), sample.name],
                capture_output=True,
                text=True,
                timeout=int(os.getenv("CLAMAV_SCAN_TIMEOUT_SECONDS", "120")),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(
                status_code=503,
                detail="The antivirus scanner is unavailable. Please try again later.",
            ) from exc
    if result.returncode == 1:
        raise HTTPException(
            status_code=422,
            detail="This file was flagged as unsafe and was not processed. Please upload another file.",
        )
    if result.returncode != 0:
        raise HTTPException(
            status_code=503,
            detail="The antivirus scan could not be completed. Please try again later.",
        )


def _set_progress(job_id: str, percent: int, stage: str, message: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job["progress"] = max(0, min(100, int(percent)))
            job["stage"] = stage
            job["message"] = message
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            job.setdefault("events", []).append({"percent": job["progress"], "stage": stage, "message": message, "at": job["updated_at"]})
            if len(job["events"]) > 80:
                job["events"] = job["events"][-80:]


def _work(job_id: str, filename: str, content: bytes, job_dir: Path) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        result = run_pipeline(job_id, filename, content, job_dir,
                              lambda percent, stage, message: _set_progress(job_id, percent, stage, message))
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["error"] = None
    except Exception as exc:
        print(
            f"\n[WORKFLOW ERROR] job_id={job_id} filename={filename!r}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
        _set_progress(job_id, 100, "failed", str(exc))
    finally:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    llama_models = llama.status()
    python_models = pipeline_model_status()
    all_active = all(item["active"] for item in llama_models.values()) and all(python_models.values())
    return {
        "status": "ok" if all_active else "degraded",
        "service": "geotagging-of-texts", "device": device(), "gpu": gpu_available(),
        "chat_model": CHAT_MODEL,
        "models": {"llama_cpp": llama_models, **python_models},
    }


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".xml", ".tei"}:
        raise HTTPException(status_code=415, detail="Supported input formats are PDF, TXT, and XML/TEI.")
    with _jobs_lock:
        if sum(1 for j in _jobs.values() if j["status"] in ("queued", "running")) >= MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="Too many documents are being processed. Please try again shortly.")
    # Read one byte past the limit so an oversized upload is rejected without buffering all of it.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if suffix in {".xml", ".tei"} and b"<!DOCTYPE" in content.upper():
        raise HTTPException(status_code=400, detail="XML files with a DOCTYPE declaration are not accepted for safety.")
    await asyncio.to_thread(_scan_upload, content)
    _purge_unsaved_runs()
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    # Never use the supplied filename as a filesystem path.
    upload_path = job_dir / f"source{suffix}"
    upload_path.write_bytes(content)
    now = datetime.now(timezone.utc).isoformat()
    queued_message = "Added to the local workflow."
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id, "filename": filename, "status": "queued", "progress": 1,
            "stage": "queued", "message": queued_message, "created_at": now,
            "updated_at": now, "events": [{"percent": 1, "stage": "queued", "message": queued_message, "at": now}],
            "result": None, "error": None,
        }
    background_tasks.add_task(_work, job_id, filename, content, job_dir)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    _checked_job_id(job_id)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            # Results survive app restarts on the mounted data volume.
            result_path = JOBS_DIR / job_id / "results.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                return {"job_id": job_id, "status": "completed", "progress": 100, "stage": "complete",
                        "message": "Saved result restored from disk.", "result": result, "error": None}
            raise HTTPException(status_code=404, detail="Job not found.")
        response = {k: job.get(k) for k in ("job_id", "filename", "status", "progress", "stage", "message", "created_at", "updated_at", "finished_at", "events", "error", "result")}
        return JSONResponse(response)


@app.get("/api/places/{place_id}")
async def lookup_place(place_id: str):
    """Resolve a ToposText identifier against the locally bundled gazetteer metadata."""
    _, records, warning = load_gazetteer()
    record = lookup_place_by_id(place_id, records)
    if not record:
        message = "That identifier was not found in the bundled ToposText metadata. You can still fill the fields manually."
        if warning:
            message += f" Gazetteer note: {warning}"
        raise HTTPException(status_code=404, detail=message)
    return {
        "id": str(record.get("id") or place_id),
        "kb_id": str(record.get("id") or place_id),
        "name": str(record.get("name") or ""),
        "description": str(record.get("description") or ""),
        "type": str(record.get("type") or "Unclassified place"),
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
        "url": str(record.get("url") or ""),
        "region": str(record.get("region") or ""),
        "country": str(record.get("country") or ""),
        "department": str(record.get("department") or ""),
        "modern_location": str(record.get("modern_location") or ""),
        "coordinate_confidence": str(record.get("coordinate_confidence") or ""),
        "wikipedia_url": str(record.get("wikipedia_url") or ""),
        "wikipedia_id": str(record.get("wikipedia_id") or ""),
        "pleiades_id": str(record.get("pleiades_id") or ""),
        "wikidata_id": str(record.get("wikidata_id") or ""),
        "toposid": str(record.get("toposid") or ""),
    }


def _review_location(value: Any) -> dict[str, Any]:
    """Copy just the supported place fields from a browser review payload."""
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Every place must be an object.")
    place_id = str(value.get("id") or "").strip()
    if not place_id or len(place_id) > 120:
        raise HTTPException(status_code=400, detail="A place needs a valid internal ID.")
    output: dict[str, Any] = {"id": place_id}
    text_fields = ("kb_id", "name", "description", "type", "url", "region", "country", "department",
                   "coordinate_confidence", "modern_location", "wikipedia_url", "wikipedia_id", "pleiades_id",
                   "wikidata_id", "toposid", "match_method")
    for key in text_fields:
        output[key] = str(value.get(key) or "")[:5000]
    for key, minimum, maximum in (("latitude", -90.0, 90.0), ("longitude", -180.0, 180.0)):
        raw = value.get(key)
        if raw in (None, ""):
            output[key] = None
        else:
            try:
                number = float(raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{key.title()} must be a number.")
            if not (minimum <= number <= maximum):
                raise HTTPException(status_code=400, detail=f"{key.title()} is outside the valid range.")
            output[key] = number
    output["mention_count"] = 0
    return output


@app.post("/api/jobs/{job_id}/review")
async def save_review(job_id: str, payload: dict[str, Any] = Body(...)):
    """Persist reviewed mentions and regenerate both JSON and TEI exports."""
    _checked_job_id(job_id)
    job_dir = JOBS_DIR / job_id
    result_path = job_dir / "results.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="The workflow result was not found on disk.")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="The saved workflow result could not be read.")

    raw_mentions = payload.get("mentions")
    raw_locations = payload.get("locations")
    if not isinstance(raw_mentions, list) or not isinstance(raw_locations, list):
        raise HTTPException(status_code=400, detail="The review payload must include mentions and locations arrays.")
    paragraphs = result.get("paragraphs") or []
    locations_by_id: dict[str, dict[str, Any]] = {}
    for raw_location in raw_locations:
        location = _review_location(raw_location)
        locations_by_id[location["id"]] = location

    mentions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    occupied: dict[int, list[tuple[int, int]]] = {}
    for raw_mention in raw_mentions:
        if not isinstance(raw_mention, dict):
            raise HTTPException(status_code=400, detail="Every mention must be an object.")
        try:
            paragraph_index = int(raw_mention.get("paragraph_index"))
            start = int(raw_mention.get("start"))
            end = int(raw_mention.get("end"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="A mention has invalid source offsets.")
        if paragraph_index < 0 or paragraph_index >= len(paragraphs):
            raise HTTPException(status_code=400, detail="A mention points to a paragraph that does not exist.")
        source_text = str(paragraphs[paragraph_index].get("text") or "")
        if start < 0 or end <= start or end > len(source_text):
            raise HTTPException(status_code=400, detail="A mention has invalid source offsets.")
        actual_text = source_text[start:end]
        requested_id = str(raw_mention.get("id") or f"m-review-{paragraph_index}-{start}")[:120]
        if requested_id in seen_ids:
            raise HTTPException(status_code=400, detail="Mention IDs must be unique.")
        seen_ids.add(requested_id)
        spans = occupied.setdefault(paragraph_index, [])
        if any(start < old_end and end > old_start for old_start, old_end in spans):
            raise HTTPException(status_code=400, detail="Two mentions overlap in the source text. Please adjust the selection.")
        spans.append((start, end))
        mention = {
            "id": requested_id, "paragraph_index": paragraph_index, "start": start, "end": end,
            "mention": actual_text,
            "label": str(raw_mention.get("label") or "PLACE")[:40],
            "recontext": str(raw_mention.get("recontext") or "")[:5000],
            "match_score": raw_mention.get("match_score"),
            "detection_method": str(raw_mention.get("detection_method") or "Manual review")[:200],
            "match_method": str(raw_mention.get("match_method") or "manual")[:100],
            "context": str(raw_mention.get("context") or source_text)[:10000],
            "snippet": str(raw_mention.get("snippet") or source_text[max(0, start - 80):min(len(source_text), end + 80)])[:10000],
        }
        raw_location = raw_mention.get("location") if isinstance(raw_mention.get("location"), dict) else {}
        location_id = str(raw_location.get("id") or "")
        if location_id not in locations_by_id:
            # Permit the location embedded in a mention for unresolved/orphaned records,
            # but validate it through the same field allowlist as explicit locations.
            if raw_location:
                nested_location = _review_location(raw_location)
                locations_by_id[nested_location["id"]] = nested_location
                location_id = nested_location["id"]
            else:
                raise HTTPException(status_code=400, detail="Every mention must have a linked place record.")
        mention["location"] = locations_by_id[location_id]
        mentions.append(mention)

    # Keep only places referenced by at least one mention, and recompute mention counts.
    used_locations: dict[str, dict[str, Any]] = {}
    for mention in mentions:
        linked = locations_by_id[mention["location"]["id"]]
        linked["mention_count"] = int(linked.get("mention_count") or 0) + 1
        used_locations[linked["id"]] = linked
        mention["location"] = linked
    locations = list(used_locations.values())
    by_id = {location["id"]: location for location in locations}
    for mention in mentions:
        mention["location"] = by_id[mention["location"]["id"]]

    result["mentions"] = mentions
    result["locations"] = locations
    result["stats"] = {
        "paragraphs": len(paragraphs),
        "mentions": len(mentions),
        "disambiguated": sum(1 for mention in mentions if mention.get("location", {}).get("url")),
        "mapped": sum(1 for location in locations if location.get("latitude") is not None and location.get("longitude") is not None),
        "unresolved": sum(1 for mention in mentions if mention.get("location", {}).get("type") == "Unresolved"),
    }
    try:
        result["tei_filename"] = write_notebook_tei(
            job_dir, str(result.get("filename") or "document"), paragraphs, mentions, locations
        )
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not save the reviewed exports: {exc}")
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["result"] = result
    return {"status": "saved", "result": result}


@app.get("/api/jobs/{job_id}/download")
async def download_tei(job_id: str):
    _checked_job_id(job_id)
    job_dir = JOBS_DIR / job_id
    result_file = job_dir / "results.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="TEI output is not ready yet.")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    relative_name = Path(str(result.get("tei_filename", "")))
    tei_path = (job_dir / relative_name).resolve()
    if not tei_path.exists() or not tei_path.is_relative_to(job_dir.resolve()) or not tei_path.is_file():
        raise HTTPException(status_code=404, detail="TEI output file was not found.")
    if tei_path.suffix.lower() == ".zip":
        return FileResponse(path=str(tei_path), filename=tei_path.name, media_type="application/zip")
    return FileResponse(path=str(tei_path), filename=tei_path.name, media_type="application/tei+xml")


@app.get("/api/jobs/{job_id}/results.json")
async def download_json(job_id: str):
    _checked_job_id(job_id)
    path = JOBS_DIR / job_id / "results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Results are not ready yet.")
    return FileResponse(path=str(path), filename=f"{job_id}_results.json", media_type="application/json")


