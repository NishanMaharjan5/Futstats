"""
FutStats API — FastAPI backend that accepts a video upload, runs the EXISTING
3-stage analytics pipeline (+ an H.264 transcode) as a background job, and serves
the futstats-ui dashboard. One process, one `uvicorn` command.

Run (use the env that has ultralytics/torch/opencv — the API shells the pipeline
scripts out as subprocesses with THIS same interpreter):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Open http://localhost:8000/  (serves futstats-ui; its default data/results.json
still shows the pre-processed demo until you upload a new video).

Per-job pipeline (each stage a subprocess; non-zero exit => job 'failed' with the
captured stderr tail — a crashed stage never hangs the job at 'processing'):
    1. "Detecting & tracking players"  src/analytics_core.py  -> raw.pkl (+ FMP4 video)
    2. "Computing stats"               src/analytics_stats.py -> stats/
    3. "Transcoding video"             ffmpeg FMP4 -> H.264   -> assets/annotated.mp4
    4. "Building dashboard data"       src/build_ui_data.py   -> results.json + assets/

Env hooks:
    FUTSTATS_MAX_UPLOAD_MB   upload size cap in MB (default 2048 = 2 GB)
    FUTSTATS_MAX_FRAMES      TEST ONLY — caps analytics_core frames for a fast smoke run

Known limitation: the job registry is IN-MEMORY, so a server restart forgets
in-flight/finished jobs (their files remain on disk under runs/api_jobs/).
"""
import os
import sys
import json
import time
import uuid
import shutil
import threading
import subprocess
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]          # .../FutStats
JOBS_DIR = REPO / "runs" / "api_jobs"
UI_DIR = REPO / "futstats-ui"
PY = sys.executable                                  # same interpreter running uvicorn
ALLOWED_EXT = {".mp4", ".mov", ".avi"}
MAX_UPLOAD_BYTES = int(os.environ.get("FUTSTATS_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
TEST_MAX_FRAMES = os.environ.get("FUTSTATS_MAX_FRAMES")  # testing hook only

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FutStats API")

JOBS = {}                       # job_id -> {status, stage, error, created, filename}
JOBS_LOCK = threading.Lock()


def _set(job_id, **kw):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)


_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _clean(s):
    """Sanitise captured stderr for JSON/UI: strip ANSI colour codes + control chars."""
    s = _ANSI.sub("", s or "")
    s = "".join(ch for ch in s if ch == "\n" or ch >= " ")
    return s.strip()


def _run(cmd, stage, job_id):
    """Run one pipeline stage as a subprocess. Raise on non-zero exit with a
    sanitised stderr tail so the job fails loudly (never hangs at 'processing')."""
    _set(job_id, stage=stage)
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = _clean(proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"[{stage}] exit {proc.returncode}: {tail}")


def run_job(job_id, video_path):
    job = JOBS_DIR / job_id
    stats, assets = job / "stats", job / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    pkl, fmp4, h264, results = job / "raw.pkl", job / "annotated_fmp4.mp4", assets / "annotated.mp4", job / "results.json"
    try:
        _set(job_id, status="processing")

        core = [PY, "src/analytics_core.py", str(video_path),
                "--out", str(pkl), "--video-out", str(fmp4), "--video-scale", "0.5"]
        if TEST_MAX_FRAMES:
            core += ["--max-frames", str(TEST_MAX_FRAMES)]
        _run(core, "Detecting & tracking players", job_id)

        _run([PY, "src/analytics_stats.py", "--in", str(pkl), "--outdir", str(stats),
              "--players-per-team", "4"], "Computing stats", job_id)

        _run(["ffmpeg", "-y", "-i", str(fmp4), "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", "-an", str(h264)],
             "Transcoding video", job_id)
        fmp4.unlink(missing_ok=True)                 # drop the big intermediate

        _run([PY, "src/build_ui_data.py",
              "--stats-dir", str(stats), "--pkl", str(pkl),
              "--assets-out", str(assets),
              "--asset-url-prefix", f"/api/jobs/{job_id}/assets/",
              "--video-url", f"/api/jobs/{job_id}/assets/annotated.mp4",
              "--out-json", str(results),
              "--team-a-name", "Team A", "--team-b-name", "Team B",
              "--venue", "Uploaded clip", "--date", time.strftime("%Y-%m-%d")],
             "Building dashboard data", job_id)

        _set(job_id, status="done", stage="Done", error=None)
    except Exception as e:                            # any stage crash -> failed, never stuck
        _set(job_id, status="failed", stage="Failed", error=str(e)[:1800])


# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload(video: UploadFile = File(...)):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported type '{ext or '?'}'. Allowed: {sorted(ALLOWED_EXT)}")

    job_id = uuid.uuid4().hex[:12]
    job = JOBS_DIR / job_id
    job.mkdir(parents=True, exist_ok=True)
    dest = job / f"input{ext}"

    size = 0
    with open(dest, "wb") as f:                       # stream to disk, enforce size cap
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                shutil.rmtree(job, ignore_errors=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
            f.write(chunk)
    if size == 0:
        shutil.rmtree(job, ignore_errors=True)
        raise HTTPException(400, "Empty file")

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "stage": "Queued", "error": None,
                        "created": time.time(), "filename": video.filename}
    threading.Thread(target=run_job, args=(job_id, dest), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
    if j:
        return {"status": j["status"], "stage": j.get("stage"), "error": j.get("error")}
    # not in the in-memory registry (e.g. server restarted) -> fall back to disk so a
    # restart mid-run doesn't strand the job as an unknown id.
    jobdir = JOBS_DIR / job_id
    if (jobdir / "results.json").exists():
        return {"status": "done", "stage": "Done", "error": None}
    if jobdir.is_dir():
        return {"status": "failed", "stage": "Interrupted",
                "error": "Job was interrupted by a server restart mid-run — please re-upload."}
    raise HTTPException(404, "unknown job_id")


@app.get("/api/results/{job_id}")
def results(job_id: str):
    rp = JOBS_DIR / job_id / "results.json"
    if rp.exists():                        # serve from disk regardless of memory (survives restart)
        return JSONResponse(json.loads(rp.read_text()))
    with JOBS_LOCK:
        j = JOBS.get(job_id)
    if j is None and not (JOBS_DIR / job_id).is_dir():
        raise HTTPException(404, "unknown job_id")
    raise HTTPException(404, f"not ready (status={j['status'] if j else 'interrupted'})")


@app.get("/api/jobs/{job_id}/assets/{filename}")
def job_asset(job_id: str, filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    p = JOBS_DIR / job_id / "assets" / filename
    if not p.is_file():
        raise HTTPException(404, "asset not found")
    return FileResponse(str(p))


# serve the dashboard (LAST so it doesn't shadow the /api/* routes above)
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
