"""
CampusGuard Vision - Backend (Phase 2: real pipeline integration)

Phase scope:
  - POST /upload            -> store uploaded video, return video_id
  - POST /start/{video_id}  -> start the real CampusGuard pipeline (GPU) as a
                                background worker that feeds an annotated
                                MJPEG frame queue + an alert queue
  - GET  /stream/{video_id} -> MJPEG of the PIPELINE'S annotated frames
  - WS   /ws/alerts          -> real fall/fight/bag alerts from the pipeline
  - POST /stop               -> stop the worker (non-blocking)

No frontend, no fake/demo alerts. Detection logic lives in backend.pipeline,
which reuses main.py verbatim.
"""

import os
import json
import uuid
import asyncio
import threading
import queue as queue_module
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline import CampusGuardPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

_event_loop = None
_uploads: dict = {}          # video_id -> {"filename": str, "path": Path}
_active: dict | None = None  # current processing session
_active_lock = threading.Lock()
_ws_connections: list = []
_ws_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    yield
    stop_active_session(blocking=False)


app = FastAPI(title="CampusGuard Vision", lifespan=lifespan)


def _broadcast_alert(alert: dict):
    if _event_loop is None:
        return
    payload = json.dumps(alert)
    with _ws_lock:
        conns = list(_ws_connections)
    for ws in conns:
        try:
            fut = asyncio.run_coroutine_threadsafe(ws.send_text(payload), _event_loop)
            fut.result(timeout=3)
        except Exception:
            with _ws_lock:
                if ws in _ws_connections:
                    _ws_connections.remove(ws)


def _flush_alerts(alert_queue, pipeline, stop_event, flush_stop):
    """Daemon thread: drain alert_queue -> push to all WebSocket clients."""
    while True:
        if flush_stop.is_set():
            break
        try:
            alert = alert_queue.get(timeout=0.5)
        except queue_module.Empty:
            if not pipeline.is_running() and alert_queue.empty():
                break
            continue
        _broadcast_alert(alert)
    # drain whatever remains after the pipeline stopped
    while True:
        try:
            alert = alert_queue.get_nowait()
        except queue_module.Empty:
            break
        _broadcast_alert(alert)


def stop_active_session(blocking=True):
    """Stop+clear the current processing session. Returns list of stopped video_ids."""
    global _active
    with _active_lock:
        session = _active
        _active = None
    if session is None:
        return []
    video_id = session.get("video_id")
    session["stop_event"].set()
    session["flush_stop"].set()
    # unblock the stream generator immediately
    fq = session.get("frame_queue")
    if fq is not None:
        try:
            while True:
                fq.get_nowait()
        except queue_module.Empty:
            pass
    stopped = []
    pipeline = session.get("pipeline")
    if pipeline is not None and pipeline.is_running():
        pipeline.stop()  # sets stop_event + bounded join(timeout=2)
        stopped.append(video_id)
    if blocking:
        ft = session.get("flush_thread")
        if ft and ft.is_alive():
            ft.join(timeout=3)
    return stopped


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    video_id = uuid.uuid4().hex
    ext = Path(file.filename).suffix or ".mp4"
    saved_path = UPLOAD_DIR / f"{video_id}{ext}"

    total = 0
    with saved_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out.close()
                saved_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file too large")
            out.write(chunk)

    _uploads[video_id] = {"filename": file.filename, "path": saved_path}
    size_mb = round(os.path.getsize(saved_path) / (1024 * 1024), 2)
    print(f"[Server] Uploaded {file.filename} -> id={video_id} ({size_mb} MB)", flush=True)
    return JSONResponse({"video_id": video_id, "filename": file.filename, "size_mb": size_mb})


@app.post("/start/{video_id}")
def start_pipeline(video_id: str):
    upload = _uploads.get(video_id)
    if upload is None:
        raise HTTPException(status_code=404, detail=f"No upload found for video_id={video_id}")

    # New session: stop+clear any previous session cleanly (no stale queues/state).
    stop_active_session(blocking=True)

    frame_queue = queue_module.Queue(maxsize=4)
    alert_queue = queue_module.Queue()
    stop_event = threading.Event()
    flush_stop = threading.Event()

    pipeline = CampusGuardPipeline(
        upload["path"], frame_queue=frame_queue, alert_queue=alert_queue, stop_event=stop_event
    )
    pipeline.start()

    flush_thread = threading.Thread(
        target=_flush_alerts,
        args=(alert_queue, pipeline, stop_event, flush_stop),
        daemon=True,
    )
    flush_thread.start()

    global _active
    with _active_lock:
        _active = {
            "video_id": video_id,
            "pipeline": pipeline,
            "frame_queue": frame_queue,
            "alert_queue": alert_queue,
            "stop_event": stop_event,
            "flush_stop": flush_stop,
            "flush_thread": flush_thread,
            "started_at": datetime.now(timezone.utc),
        }
    print(f"[Server] Started pipeline for {video_id}", flush=True)
    return JSONResponse({"video_id": video_id, "status": "started"})


@app.post("/stop")
def stop_pipeline():
    # Non-blocking: signal the worker to stop and return immediately.
    stopped = stop_active_session(blocking=False)
    print(f"[Server] Stop acknowledged for: {stopped}", flush=True)
    return JSONResponse({"stopped": stopped})


@app.get("/debug/status")
def debug_status():
    with _active_lock:
        s = _active
    if s is None:
        return JSONResponse({"active": False})
    p = s["pipeline"]
    return JSONResponse({
        "active": True,
        "video_id": s["video_id"],
        "running": p.is_running(),
        "processed_frames": p.processed_frames,
        "queued_alerts": list(p.alerts),
    })


@app.get("/stream/{video_id}")
def stream_mjpeg(video_id: str):
    with _active_lock:
        session = _active
    if session is None or session["video_id"] != video_id:
        raise HTTPException(status_code=404, detail=f"No active session for {video_id}")

    pipeline = session["pipeline"]
    frame_queue = session["frame_queue"]

    def _generate():
        try:
            while True:
                if not pipeline.is_running() and frame_queue.empty():
                    break
                try:
                    frame = frame_queue.get(timeout=0.5)
                except queue_module.Empty:
                    continue
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                data = buf.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                )
        except GeneratorExit:
            pass

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws.accept()
    with _ws_lock:
        _ws_connections.append(ws)
    print(f"[Server] WS client connected ({len(_ws_connections)} total)", flush=True)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_connections:
                _ws_connections.remove(ws)
        try:
            await ws.close()
        except Exception:
            pass
        print(f"[Server] WS client disconnected ({len(_ws_connections)} remaining)", flush=True)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static_frontend")
