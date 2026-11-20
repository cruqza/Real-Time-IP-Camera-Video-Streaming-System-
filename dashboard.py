# Browser dashboard for the IP camera viewer (Flask).
import atexit
import os
import sys
import threading
import time

import cv2
from flask import Flask, Response, abort, jsonify, render_template_string, request, send_from_directory

from camerap import (
    RECORDING_DIR,
    SCREENSHOT_DIR,
    YOLO_MODEL,
    Camera,
    ObjectDetector,
    _load_cameras,
    _tile_width,
    _validate_url,
)

HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")   # 0.0.0.0 = reachable from other devices on the Wi-Fi
PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
STREAM_FPS = 15          # max frames per second sent to each browser
JPEG_QUALITY = 80        # 0-100, lower = less bandwidth


class Dashboard:
    """Owns the cameras and a single processing thread shared by every browser."""

    def __init__(self, configured):
        print(f"Loading YOLO model {YOLO_MODEL}...")
        self.yolo = ObjectDetector()
        print("YOLO ready." if self.yolo.available
              else f"Object detection disabled: {self.yolo.error}")

        width = _tile_width(len(configured))
        self.cameras = [Camera(i, name, url, width, self.yolo.available)
                        for i, (name, url) in enumerate(configured, start=1)]
        self.lock = threading.Lock()          # guards camera state (processing vs. button presses)
        self.jpegs = {}                       # camera number -> latest JPEG bytes
        self.messages = []                    # recent events for the browser log
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        for cam in self.cameras:
            print(f"Connecting to {cam.name} at {cam.stream.url}...")
            cam.stream.start()
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.thread.join(timeout=2.0)
        with self.lock:
            for cam in self.cameras:
                cam.close()               # finishes recordings, stops stream threads
        print("Dashboard stopped, cameras released.")

    def log(self, text):
        print(text)
        self.messages.append({"time": time.strftime("%H:%M:%S"), "text": text})
        del self.messages[:-20]       # keep the last 20

    # -- processing loop ---------------------------------------------------
    def _loop(self):
        encode = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        while self.running:
            with self.lock:
                for cam in self.cameras:
                    try:
                        note = cam.update(self.yolo)
                    except Exception as exc:          # one bad frame never kills the dashboard
                        note = f"{cam.name}: processing error ({exc})"
                    if note:
                        self.log(note)
                    if cam.yolo_error is not None:
                        self.log(f"{cam.name}: YOLO error, object detection turned off")
                        cam.yolo_error = None
                    if cam.stream.status == "lost" and not cam.lost_reported:
                        cam.lost_reported = True
                        self.log(f"{cam.name}: {cam.stream.error}")
                        if cam.recording:
                            cam.stop_recording()

                    # The browser gets the same tile the desktop window shows (no border).
                    tile = cam.render(self.yolo.available, selected=False)
                    ok, buf = cv2.imencode(".jpg", tile, encode)
                    if ok:
                        self.jpegs[cam.number] = buf.tobytes()
            time.sleep(0.01)

    # -- controls (called from Flask request threads) -----------------------
    def control(self, target, action):
        """Apply a button press. target 0 = all cameras, else one camera number."""
        with self.lock:
            if target == 0:
                targets = self.cameras
            elif 1 <= target <= len(self.cameras):
                targets = [self.cameras[target - 1]]
            else:
                return False, "Unknown camera"
            live = [c for c in targets if c.tile is not None]
            if action in ("screenshot", "record") and not live:
                return False, "No live camera to use"

            if action == "screenshot":
                saved = [p for p in (c.screenshot() for c in live) if p]
                text = (f"{len(saved)} screenshot(s) saved" if saved else "Screenshot failed")
            elif action == "record":
                if any(c.recording for c in live):
                    for c in live:
                        if c.recording:
                            c.stop_recording()
                    text = "Recording saved"
                else:
                    ok = [c for c in live if c.start_recording("manual")]
                    text = f"Recording started on {len(ok)} camera(s)" if ok else "Recording failed"
            elif action == "motion":
                on = not all(c.motion_enabled for c in targets)
                for c in targets:
                    c.motion_enabled = on
                    c.motion.reset()
                text = f"Motion detection {'ON' if on else 'OFF'}"
            elif action == "yolo":
                if not self.yolo.available:
                    return False, "Object detection unavailable: pip install ultralytics"
                on = not all(c.yolo_enabled for c in targets)
                for c in targets:
                    c.yolo_enabled = on
                    c.detections = []
                text = f"Object detection {'ON' if on else 'OFF'}"
            elif action == "auto":
                on = not all(c.auto_enabled for c in targets)
                for c in targets:
                    if not on and c.rec_mode == "auto":
                        c.stop_recording()
                    c.auto_enabled = on
                    c.auto.cancel()
                    c.preroll.clear()
                text = f"Auto-recording {'armed' if on else 'OFF'}"
            else:
                return False, f"Unknown action '{action}'"

        scope = "All cameras" if target == 0 else self.cameras[target - 1].name
        self.log(f"{scope}: {text}")
        return True, text

    def status(self):
        with self.lock:
            cams = [{
                "number": c.number,
                "name": c.name,
                "status": c.stream.status,
                "fps": round(c.fps.fps, 1),
                "recording": c.recording,
                "rec_mode": c.rec_mode,
                "rec_seconds": int(time.time() - c.rec_start) if c.recording else 0,
                "motion": c.motion_enabled,
                "moving": bool(c.boxes),
                "yolo": c.yolo_enabled,
                "objects": len(c.detections) if c.yolo_enabled else 0,
                "auto": c.auto_enabled,
            } for c in self.cameras]
        return {"cameras": cams, "yolo_available": self.yolo.available,
                "messages": self.messages[-8:]}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
dashboard = None   # created in main()

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IP Camera Dashboard</title>
<style>
  :root { --bg:#111418; --card:#1b2027; --line:#2a313b; --text:#e6e9ee; --dim:#8b95a3;
          --green:#2ecc71; --red:#e74c3c; --yellow:#f1c40f; --blue:#3498db; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--text); }
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:14px 20px;
           border-bottom:1px solid var(--line); }
  header h1 { font-size:18px; margin:0 12px 0 0; }
  .grid { display:grid; gap:16px; padding:16px 20px;
          grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .card img { width:100%; display:block; background:#000; min-height:200px; }
  .bar { display:flex; align-items:center; justify-content:space-between; padding:10px 12px; gap:8px; flex-wrap:wrap; }
  .name { font-weight:600; }
  .pills span { font-size:12px; padding:2px 8px; border-radius:20px; background:#2a313b; color:var(--dim); margin-left:4px; }
  .pills .live { background:rgba(46,204,113,.15); color:var(--green); }
  .pills .lost { background:rgba(231,76,60,.15); color:var(--red); }
  .pills .rec  { background:var(--red); color:#fff; }
  .pills .move { background:rgba(241,196,15,.18); color:var(--yellow); }
  .controls { display:flex; gap:6px; flex-wrap:wrap; padding:0 12px 12px; }
  button { background:#2a313b; color:var(--text); border:1px solid var(--line); border-radius:6px;
           padding:7px 12px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--blue); }
  button.on { background:rgba(52,152,219,.2); border-color:var(--blue); }
  button.recording { background:var(--red); border-color:var(--red); color:#fff; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .panels { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:0 20px 24px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .panel h2 { font-size:14px; margin:0 0 8px; color:var(--dim); font-weight:600; }
  .panel ul { list-style:none; margin:0; padding:0; max-height:220px; overflow:auto; font-size:13px; }
  .panel li { padding:4px 0; border-bottom:1px solid var(--line); }
  .panel a { color:var(--blue); text-decoration:none; }
  .toast { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:#000c; padding:10px 16px;
           border-radius:8px; font-size:14px; opacity:0; transition:opacity .3s; pointer-events:none; }
  .toast.show { opacity:1; }
  @media (max-width:700px) { .panels { grid-template-columns:1fr; } .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>📷 IP Camera Dashboard</h1>
  <span style="color:var(--dim);font-size:13px">All cameras:</span>
  <button onclick="send(0,'screenshot')">Screenshot all</button>
  <button onclick="send(0,'record')">Record all</button>
  <button onclick="send(0,'motion')">Motion all</button>
  <button onclick="send(0,'yolo')" {{ '' if yolo else 'disabled' }}>YOLO all</button>
  <button onclick="send(0,'auto')">Auto-rec all</button>
</header>

<div class="grid">
{% for cam in cameras %}
  <div class="card" id="cam{{ cam.number }}">
    <img src="{{ url_for('video', number=cam.number) }}" alt="{{ cam.name }} stream">
    <div class="bar">
      <span class="name">[{{ cam.number }}] {{ cam.name }}</span>
      <span class="pills"></span>
    </div>
    <div class="controls">
      <button data-a="screenshot" onclick="send({{ cam.number }},'screenshot')">📸 Screenshot</button>
      <button data-a="record" onclick="send({{ cam.number }},'record')">⏺ Record</button>
      <button data-a="motion" onclick="send({{ cam.number }},'motion')">Motion</button>
      <button data-a="yolo" onclick="send({{ cam.number }},'yolo')" {{ '' if yolo else 'disabled' }}>YOLO</button>
      <button data-a="auto" onclick="send({{ cam.number }},'auto')">Auto-rec</button>
    </div>
  </div>
{% endfor %}
</div>

<div class="panels">
  <div class="panel"><h2>Activity</h2><ul id="log"></ul></div>
  <div class="panel"><h2>Saved files</h2><ul id="files"></ul></div>
</div>
<div class="toast" id="toast"></div>

<script>
function toast(t){ const el=document.getElementById('toast'); el.textContent=t; el.classList.add('show');
  clearTimeout(el._t); el._t=setTimeout(()=>el.classList.remove('show'),2000); }

async function send(camera, action){
  const r = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'},
                                          body: JSON.stringify({camera, action})});
  const j = await r.json(); toast(j.message); refresh(); setTimeout(loadFiles, 800);
}

function mmss(s){ return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0'); }

async function refresh(){
  try {
    const j = await (await fetch('/api/status')).json();
    for (const c of j.cameras){
      const card = document.getElementById('cam'+c.number); if(!card) continue;
      let p = `<span class="${c.status==='live'?'live':c.status==='lost'?'lost':''}">${c.status}</span>`
            + `<span>${c.fps} fps</span>`;
      if (c.moving) p += `<span class="move">motion</span>`;
      if (c.yolo)   p += `<span>${c.objects} objects</span>`;
      if (c.recording) p += `<span class="rec">● ${c.rec_mode==='auto'?'AUTO ':''}REC ${mmss(c.rec_seconds)}</span>`;
      card.querySelector('.pills').innerHTML = p;
      const b = a => card.querySelector(`button[data-a="${a}"]`);
      b('record').classList.toggle('recording', c.recording);
      b('record').textContent = c.recording ? '⏹ Stop' : '⏺ Record';
      b('motion').classList.toggle('on', c.motion);
      b('yolo').classList.toggle('on', c.yolo);
      b('auto').classList.toggle('on', c.auto);
    }
    document.getElementById('log').innerHTML =
      j.messages.slice().reverse().map(m=>`<li>${m.time} — ${m.text}</li>`).join('') || '<li>No activity yet</li>';
  } catch(e) {}
}

async function loadFiles(){
  try {
    const j = await (await fetch('/api/files')).json();
    const rows = [...j.recordings.map(f=>['recordings',f]), ...j.screenshots.map(f=>['screenshots',f])];
    document.getElementById('files').innerHTML = rows.length
      ? rows.map(([k,f])=>`<li><a href="/files/${k}/${encodeURIComponent(f)}" download>${f}</a></li>`).join('')
      : '<li>Nothing saved yet</li>';
  } catch(e) {}
}

setInterval(refresh, 1000); setInterval(loadFiles, 5000); refresh(); loadFiles();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, cameras=dashboard.cameras, yolo=dashboard.yolo.available)


def _mjpeg(number):
    """Yield the camera's latest JPEG as a multipart (MJPEG) stream."""
    delay = 1.0 / STREAM_FPS
    last = None
    while dashboard.running:
        jpg = dashboard.jpegs.get(number)
        if jpg is not None and jpg is not last:
            last = jpg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        time.sleep(delay)


@app.route("/video/<int:number>")
def video(number):
    if not 1 <= number <= len(dashboard.cameras):
        abort(404)
    return Response(_mjpeg(number), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(dashboard.status())


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(silent=True) or {}
    try:
        target = int(data.get("camera", 0))
    except (TypeError, ValueError):
        return jsonify(ok=False, message="camera must be a number"), 400
    ok, text = dashboard.control(target, str(data.get("action", "")))
    return jsonify(ok=ok, message=text), (200 if ok else 400)


def _list_files(folder, exts):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)[:30]


@app.route("/api/files")
def api_files():
    return jsonify(recordings=_list_files(RECORDING_DIR, (".mp4",)),
                   screenshots=_list_files(SCREENSHOT_DIR, (".jpg", ".png")))


@app.route("/files/<kind>/<path:name>")
def download(kind, name):
    folder = {"recordings": RECORDING_DIR, "screenshots": SCREENSHOT_DIR}.get(kind)
    if folder is None:
        abort(404)
    # send_from_directory refuses paths that escape the folder (e.g. ../)
    return send_from_directory(folder, name, as_attachment=True)


def main():
    global dashboard
    configured = _load_cameras()
    if not configured:
        print("Error: camera address not configured. Set CAMERA_URL (or CAMERA_URLS) "
              "in your environment or in a .env file next to camerap.py.")
        sys.exit(1)
    for name, url in configured:
        err = _validate_url(url)
        if err:
            print(f"Error ({name}): {err}")
            sys.exit(1)

    dashboard = Dashboard(configured)
    atexit.register(dashboard.stop)      # safe shutdown on Ctrl+C or normal exit
    dashboard.start()

    print(f"Dashboard running: open http://localhost:{PORT} in your browser "
          f"(Ctrl+C to stop)")
    try:
        # threaded=True: each browser stream gets its own thread.
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    except OSError as exc:
        print(f"Error: could not start the web server on port {PORT} ({exc}). "
              "Set DASHBOARD_PORT to a free port.")
    finally:
        dashboard.stop()


if __name__ == "__main__":
    main()