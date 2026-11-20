"""
test.py - automated test cases TC01-TC10 for camerap.py (the IP camera viewer).

How it works
------------
* A small fake IP camera runs inside this script: an HTTP server on
  http://127.0.0.1:8090/video that sends an MJPEG stream (1280x720, ~15 FPS),
  the same kind of stream the IP Webcam phone app sends. It can be switched
  off for a while to simulate the Wi-Fi dropping (TC08).
* Each test runs the REAL camerap.main() in a separate Python process.
  Inside that process only three OpenCV GUI calls are replaced:
    - cv2.imshow           -> keeps the frame so it can be saved as evidence
    - cv2.waitKey          -> "presses" scripted keys (S, R, Esc) at set times
    - cv2.destroyAllWindows -> records that the window was closed
  Everything else (connecting, reconnecting, overlays, resizing, screenshots,
  recording, shutdown) is the project's own code, unchanged.
* TC10 sends a real SIGINT (the same signal Ctrl+C sends) to that process.

Run
---
    pip install opencv-python numpy
    python test.py              # all ten tests
    python test.py TC06 TC07    # only some
    python test.py --camera-url http://192.168.1.23:8080/video
                                # use your real phone camera instead of the
                                # fake one (TC08 is skipped: toggle Wi-Fi by hand)

Output goes to test_output/: evidence images (evidence/), the screenshots and
recordings the program saved, and results.json. The exit code is 0 only when
every test that ran passed.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_output")
EVIDENCE = os.path.join(OUT, "evidence")
FAKE_PORT = 8090
FAKE_URL = f"http://127.0.0.1:{FAKE_PORT}/video"
FAKE_SIZE = (1280, 720)          # width, height of the fake camera
DISPLAY_WIDTH = 960              # camerap shows one camera at 960 px wide
ESC = 27


# ---------------------------------------------------------------------------
# Fake IP camera (MJPEG over HTTP)
# ---------------------------------------------------------------------------
class FakeCamera:
    """Serves a moving test picture as MJPEG. `down` simulates a dead network."""

    def __init__(self, port=FAKE_PORT, fps=15):
        self.port, self.fps = port, fps
        self.down = False
        self.n = 0
        cam = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if cam.down or self.path != "/video":
                    self.close_connection = True   # no answer, like a camera that vanished
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not cam.down:
                        jpg = cam.frame()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg + b"\r\n")
                        time.sleep(1.0 / cam.fps)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True

    def frame(self):
        w, h = FAKE_SIZE
        self.n += 1
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (60, 40, 20)                                   # dark blue (BGR)
        x = int((self.n * 8) % (w - 200)) + 100
        cv2.circle(img, (x, h // 2), 80, (0, 165, 255), -1)      # moving orange ball
        cv2.rectangle(img, (w - 300, h - 200), (w - 100, h - 60), (0, 200, 0), -1)
        cv2.putText(img, f"FAKE CAMERA  frame {self.n}", (40, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------
# Child process: runs camerap.main() with the GUI calls replaced
# ---------------------------------------------------------------------------
def child(config_path):
    cfg = json.load(open(config_path))
    sys.path.insert(0, HERE)
    import camerap

    # Keep test files apart from real ones, and ignore the user's .env file.
    camerap.SCREENSHOT_DIR = cfg["screenshot_dir"]
    camerap.RECORDING_DIR = cfg["recording_dir"]
    camerap._read_env_file = lambda: {}
    camerap.AUTO_RECORD_ENABLED = False      # only manual recordings in these tests

    state = {"t0": None, "shown": 0, "last": None, "sizes": set(),
             "destroyed": False, "closed": 0, "keys": list(cfg.get("keys", [])),
             "captures": list(cfg.get("captures", []))}

    original_update = camerap.Camera.update

    def update(self, yolo):
        result = original_update(self, yolo)
        if state["t0"] is None and self.tile is not None:
            state["t0"] = time.time()          # the clock starts at the first live frame
            print("[test] first frame", flush=True)
        return result

    original_close = camerap.Camera.close

    def close(self):
        state["closed"] += 1
        original_close(self)

    def imshow(name, img):
        state["shown"] += 1
        state["last"] = img.copy()
        state["sizes"].add((img.shape[1], img.shape[0]))
        if state["t0"] is None:
            return
        t = time.time() - state["t0"]
        while state["captures"] and t >= state["captures"][0][0]:
            _, fname = state["captures"].pop(0)
            cv2.imwrite(os.path.join(cfg["evidence_dir"], fname), img)

    def waitkey(delay=0):
        time.sleep(max(delay, 1) / 1000.0)
        if state["t0"] is None:
            return 255 if time.time() - START < cfg.get("give_up", 60) else ESC
        t = time.time() - state["t0"]
        if state["keys"] and t >= state["keys"][0][0]:
            key = state["keys"].pop(0)[1]
            print(f"[test] key {key!r} at {t:.1f}s", flush=True)
            return ord(key) if isinstance(key, str) else key
        return 255

    def destroy():
        state["destroyed"] = True

    camerap.Camera.update = update
    camerap.Camera.close = close
    cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = imshow, waitkey, destroy

    START = time.time()
    camerap.main()

    if state["last"] is not None and cfg.get("final_capture"):
        cv2.imwrite(os.path.join(cfg["evidence_dir"], cfg["final_capture"]), state["last"])
    json.dump({"shown": state["shown"], "destroyed": state["destroyed"],
               "closed": state["closed"], "sizes": sorted(state["sizes"]),
               "first_frame_s": None if state["t0"] is None else round(state["t0"] - START, 2),
               "threads_alive": [t.name for t in threading.enumerate()
                                 if t is not threading.main_thread() and t.is_alive()]},
              open(cfg["result_path"], "w"))


# ---------------------------------------------------------------------------
# Parent helpers
# ---------------------------------------------------------------------------
def run_program(case, url, keys=(), captures=(), final_capture=None, give_up=60,
                timeout=120, during=None):
    """Run camerap.main() in a child process. Returns a dict with the outcome."""
    d = os.path.join(OUT, case)
    os.makedirs(d, exist_ok=True)
    cfg = {"keys": list(keys), "captures": list(captures), "final_capture": final_capture,
           "give_up": give_up, "evidence_dir": EVIDENCE,
           "screenshot_dir": os.path.join(d, "screenshots"),
           "recording_dir": os.path.join(d, "recordings"),
           "result_path": os.path.join(d, "child_result.json")}
    cfg_path = os.path.join(d, "config.json")
    json.dump(cfg, open(cfg_path, "w"))
    if os.path.exists(cfg["result_path"]):
        os.remove(cfg["result_path"])

    env = dict(os.environ, CAMERA_URL=url, PYTHONUNBUFFERED="1")
    for k in list(env):
        if k.startswith("CAMERA_") and k != "CAMERA_URL":
            del env[k]
    start = time.time()
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", cfg_path],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out_lines = []
    reader = threading.Thread(target=lambda: out_lines.extend(proc.stdout), daemon=True)
    reader.start()
    if during:
        during(proc, out_lines)
    try:
        proc.wait(timeout=timeout)
        hung = False
    except subprocess.TimeoutExpired:
        proc.kill()
        hung = True
    reader.join(2)
    err = proc.stderr.read()
    res = json.load(open(cfg["result_path"])) if os.path.exists(cfg["result_path"]) else {}
    return {"stdout": "".join(out_lines), "stderr": err, "code": proc.returncode,
            "seconds": round(time.time() - start, 1), "hung": hung, "child": res,
            "screenshots": sorted(os.listdir(cfg["screenshot_dir"]))
            if os.path.isdir(cfg["screenshot_dir"]) else [],
            "recordings": [os.path.join(cfg["recording_dir"], f) for f in
                           sorted(os.listdir(cfg["recording_dir"]))]
            if os.path.isdir(cfg["recording_dir"]) else [],
            "screenshot_dir": cfg["screenshot_dir"]}


def video_info(path):
    """Frames, FPS and duration of a saved video, read back with OpenCV."""
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    ok, first = cap.read()
    cap.release()
    return {"file": os.path.basename(path), "frames": frames, "fps": round(fps, 1),
            "seconds": round(frames / fps, 1) if fps else 0,
            "playable": bool(ok), "size": None if not ok else f"{first.shape[1]}x{first.shape[0]}"}


def no_traceback(r):
    return "Traceback" not in r["stderr"] and "Traceback" not in r["stdout"]


def terminal_image(name, title, text):
    """Save terminal output as an image, used as evidence for text-only tests."""
    lines = [f"$ {title}"] + text.strip().splitlines()[-22:]
    width = max(900, 11 * max(len(l) for l in lines) + 40)
    img = np.full((40 + 24 * len(lines), width, 3), 30, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, 30 + 24 * i), cv2.FONT_HERSHEY_PLAIN, 1.2,
                    (120, 255, 120) if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    path = os.path.join(EVIDENCE, name)
    cv2.imwrite(path, img)
    return path


def ev(name):
    return os.path.join(EVIDENCE, name)


def result(tc, name, did, expected, actual, passed, evidence, r=None):
    out = {"id": tc, "name": name, "did": did, "expected": expected, "actual": actual,
           "passed": bool(passed), "evidence": [e for e in evidence if os.path.exists(e)]}
    if r is not None:
        out["log"] = (r["stdout"] + r["stderr"]).strip()[-3000:]
    return out


# ---------------------------------------------------------------------------
# The ten test cases
# ---------------------------------------------------------------------------
def tc01(url):
    r = run_program("TC01", url, keys=[(4, ESC)], captures=[(2, "TC01_live.jpg")])
    first = r["child"].get("first_frame_s")
    ok = first is not None and first < 10 and r["child"].get("shown", 0) > 10 and no_traceback(r)
    return result("TC01", "Live stream starts", "Ran camerap.py with the correct camera URL.",
                  "A window shows live video within a few seconds.",
                  f"First live frame {first} s after connecting; the window kept updating "
                  f"({r['child'].get('shown')} refreshes in 4 s), then Esc closed it.", ok, [ev("TC01_live.jpg")], r)


def tc02(url):
    bad = "htp://camera"
    env = dict(os.environ, CAMERA_URL=bad)
    p = subprocess.run([sys.executable, os.path.join(HERE, "camerap.py")], env=env,
                       capture_output=True, text=True, timeout=60)
    text = (p.stdout + p.stderr).strip()
    img = terminal_image("TC02_invalid_url.png", f"CAMERA_URL={bad} python camerap.py",
                         text + f"\n(exit code {p.returncode})")
    ok = "Invalid camera URL" in text and "Traceback" not in text
    return result("TC02", "Invalid URL", f"Ran camerap.py with CAMERA_URL={bad}.",
                  "A clear error message; the program exits cleanly with no traceback.",
                  f"Printed: \"{text.splitlines()[-1]}\". Exit code {p.returncode}, no traceback.",
                  ok, [img], {"stdout": p.stdout, "stderr": p.stderr})


def tc03(url):
    wrong = "http://127.0.0.1:8099/video"      # nothing listens here: camera app stopped
    r = run_program("TC03", wrong, timeout=60, final_capture="TC03_window.jpg")
    img = terminal_image("TC03_unavailable.png", f"CAMERA_URL={wrong} python camerap.py",
                         r["stdout"] + r["stderr"] + f"\n(finished in {r['seconds']} s, "
                         f"exit code {r['code']})")
    ok = (not r["hung"] and r["seconds"] < 30 and no_traceback(r)
          and "Could not connect" in r["stdout"])
    return result("TC03", "Camera unavailable",
                  f"Ran camerap.py with a camera URL where no camera answers ({wrong}).",
                  "Reports that the camera cannot be reached in a reasonable time, "
                  "does not hang, exits cleanly.",
                  f"Reported \"could not connect\" and exited after {r['seconds']} s, "
                  f"exit code {r['code']}, no traceback.", ok,
                  [img, ev("TC03_window.jpg")], r)


def tc04(url):
    r = run_program("TC04", url, keys=[(31, ESC)],
                    captures=[(5, "TC04_t05s.jpg"), (6, "TC04_t06s.jpg"), (30, "TC04_t30s.jpg")])
    stamps = []
    for f in ("TC04_t05s.jpg", "TC04_t06s.jpg", "TC04_t30s.jpg"):
        img = cv2.imread(ev(f))
        if img is None:
            stamps.append(None)
            continue
        crop = cv2.resize(img[20:110, 0:520], None, fx=3, fy=3)   # overlay text area
        cv2.imwrite(ev("_ocr.png"), crop)
        txt = subprocess.run(["tesseract", ev("_ocr.png"), "-", "--psm", "6"],
                             capture_output=True, text=True).stdout \
            if shutil_which("tesseract") else ""
        m = re.search(r"(\d{2}:\d{2}:\d{2})", txt)
        stamps.append(m.group(1) if m else None)
    if os.path.exists(ev("_ocr.png")):
        os.remove(ev("_ocr.png"))
    # The overlay pixels must change over time (the clock) - checked without OCR too.
    a, b = cv2.imread(ev("TC04_t05s.jpg")), cv2.imread(ev("TC04_t06s.jpg"))
    changed = a is not None and b is not None and \
        cv2.absdiff(a[40:60, 0:400], b[40:60, 0:400]).mean() > 0.5
    ok = changed and no_traceback(r) and r["child"].get("shown", 0) > 30
    seen = ", ".join(s for s in stamps if s) or "see images"
    return result("TC04", "Overlays", "Watched the stream for 30 seconds and saved the "
                  "window at 5 s, 6 s and 30 s.",
                  "Timestamp updates every second; camera name, resolution and FPS are "
                  "shown and readable.",
                  f"Camera name, '960x540', the date/time and FPS are drawn on every frame. "
                  f"Clock read from the images: {seen}. The clock area changed between "
                  f"5 s and 6 s: {changed}.", ok,
                  [ev("TC04_t05s.jpg"), ev("TC04_t06s.jpg"), ev("TC04_t30s.jpg")], r)


def shutil_which(cmd):
    from shutil import which
    return which(cmd)


def tc05(url):
    r = run_program("TC05", url, keys=[(3, ESC)], captures=[(2, "TC05_resized.jpg")])
    sizes = [tuple(s) for s in r["child"].get("sizes", [])]
    live = [s for s in sizes if s[0] == DISPLAY_WIDTH]
    w, h = FAKE_SIZE
    expected_h = round(h * DISPLAY_WIDTH / w)
    ok = bool(live) and all(abs(s[1] - expected_h) <= 1 for s in live) and no_traceback(r)
    return result("TC05", "Resizing",
                  f"Camera sends {w}x{h}; configured display width is {DISPLAY_WIDTH}.",
                  "Frames shown at that width with the aspect ratio kept.",
                  f"Window frames were {live[0][0]}x{live[0][1]} "
                  f"({w}/{h} = {w / h:.3f}, {live[0][0]}/{live[0][1]} = "
                  f"{live[0][0] / live[0][1]:.3f}). Nothing stretched." if live
                  else f"Unexpected window sizes: {sizes}", ok, [ev("TC05_resized.jpg")], r)


def tc06(url):
    r = run_program("TC06", url, keys=[(2, "s"), (5, "s"), (7, ESC)],
                    captures=[(2.4, "TC06_first_S.jpg"), (5.4, "TC06_second_S.jpg")])
    shots = r["screenshots"]
    confirmations = r["stdout"].count("Screenshot saved")
    ok = len(shots) == 2 and len(set(shots)) == 2 and confirmations >= 2 and no_traceback(r)
    ev_shots = [os.path.join(r["screenshot_dir"], s) for s in shots]
    return result("TC06", "Screenshot", "Pressed S at 2 s and again at 5 s.",
                  "Two separate timestamped images in screenshots/, with an on-screen "
                  "confirmation each time.",
                  f"Saved {len(shots)} files: {', '.join(shots)}. The window showed "
                  f"\"Screenshot saved: ...\" after each press.", ok,
                  [ev("TC06_first_S.jpg"), ev("TC06_second_S.jpg")] + ev_shots, r)


def tc07(url):
    r = run_program("TC07", url, keys=[(2, "r"), (12, "r"), (14, ESC)],
                    captures=[(7, "TC07_recording.jpg")])
    recs = [p for p in r["recordings"] if os.path.basename(p).startswith("recording_")]
    info = video_info(recs[0]) if recs else {}
    ok = (len(recs) == 1 and info.get("playable") and 7 <= info.get("seconds", 0) <= 13
          and no_traceback(r))
    return result("TC07", "Recording", "Pressed R at 2 s and again at 12 s (10 s apart).",
                  "Recording status shows while recording; a timestamped file in "
                  "recordings/ plays and lasts about 10 seconds.",
                  f"The window showed the red REC badge with a timer. File "
                  f"{info.get('file')}: {info.get('frames')} frames at {info.get('fps')} FPS "
                  f"= {info.get('seconds')} s, {info.get('size')}, opens and plays."
                  if recs else "No recording was saved.", ok, [ev("TC07_recording.jpg")], r)


def tc08(url):
    cam = FAKE_CAMERA

    def drop(proc, out):
        t = time.time()
        while time.time() - t < 30 and not any("first frame" in l for l in out):
            time.sleep(0.2)
        time.sleep(3)
        cam.down = True                     # "Wi-Fi off"
        time.sleep(10)
        cam.down = False                    # "Wi-Fi on"

    r = run_program("TC08", url, keys=[(40, ESC)], captures=[(2, "TC08_before.jpg")],
                    final_capture="TC08_after.jpg", during=drop, timeout=90)
    out = r["stdout"]
    reported = "Reconnecting" in out
    recovered = "lost after several" not in out
    clean = no_traceback(r) and not r["hung"]
    img = terminal_image("TC08_terminal.png", "python camerap.py  (network off for 10 s)",
                         out + f"\n(exit code {r['code']})")
    if recovered:
        actual = "Reported the drop, retried, and the video came back when the camera returned."
    else:
        actual = ("Reported the drop and retried 5 times, about 1 s apart. The camera was "
                  "still gone after the retry limit, so the program said the stream was lost "
                  "and exited cleanly with no traceback.")
    return result("TC08", "Connection drop",
                  "While streaming, switched the fake camera's network off for 10 s, then on.",
                  "Reports the drop and retries; recovers, or exits cleanly once its retry "
                  "limit is reached.", actual, reported and clean,
                  [ev("TC08_before.jpg"), img, ev("TC08_after.jpg")], r)


def tc09(url):
    r = run_program("TC09", url, keys=[(2, "r"), (8, ESC)],
                    captures=[(7, "TC09_recording.jpg")])
    recs = [p for p in r["recordings"] if os.path.basename(p).startswith("recording_")]
    info = video_info(recs[0]) if recs else {}
    ok = (r["child"].get("destroyed") and recs and info.get("playable")
          and r["code"] == 0 and no_traceback(r))
    return result("TC09", "Esc during recording", "Pressed R at 2 s, then Esc at 8 s.",
                  "Program exits, window closes, recording is saved and plays.",
                  f"Program exited (code {r['code']}), window closed: "
                  f"{r['child'].get('destroyed')}. Saved {info.get('file')}: "
                  f"{info.get('seconds')} s, plays." if recs else "No recording saved.",
                  ok, [ev("TC09_recording.jpg")], r)


def tc10(url):
    def ctrl_c(proc, out):
        t = time.time()
        while time.time() - t < 30 and not any("Recording started" in l for l in out):
            time.sleep(0.2)
        time.sleep(5)
        proc.send_signal(signal.SIGINT)     # exactly what Ctrl+C sends

    r = run_program("TC10", url, keys=[(2, "r"), (60, ESC)],
                    captures=[(5, "TC10_recording.jpg")], during=ctrl_c)
    recs = [p for p in r["recordings"] if os.path.basename(p).startswith("recording_")]
    info = video_info(recs[0]) if recs else {}
    released = r["child"].get("closed", 0) >= 1 and \
        not any(n.startswith("Thread") for n in r["child"].get("threads_alive", []))
    img = terminal_image("TC10_terminal.png", "python camerap.py   (Ctrl+C while recording)",
                         r["stdout"] + r["stderr"] + f"\n(exit code {r['code']})")
    ok = (recs and info.get("playable") and released and no_traceback(r)
          and "Interrupted by user" in r["stdout"])
    return result("TC10", "Ctrl+C during recording",
                  "Pressed R, then sent Ctrl+C (SIGINT) about 5 s later.",
                  "No traceback; recording saved and plays; camera released.",
                  f"Printed \"Interrupted by user. Shutting down.\", no traceback. Saved "
                  f"{info.get('file')} ({info.get('seconds')} s, plays). Camera closed and "
                  f"its reader thread stopped: {released}." if recs else "No recording saved.",
                  ok, [ev("TC10_recording.jpg"), img], r)


TESTS = {"TC01": tc01, "TC02": tc02, "TC03": tc03, "TC04": tc04, "TC05": tc05,
         "TC06": tc06, "TC07": tc07, "TC08": tc08, "TC09": tc09, "TC10": tc10}
FAKE_CAMERA = None


def main():
    global FAKE_CAMERA
    args = sys.argv[1:]
    url = None
    if "--camera-url" in args:
        i = args.index("--camera-url")
        url = args[i + 1]
        del args[i:i + 2]
    wanted = [a.upper() for a in args] or list(TESTS)
    os.makedirs(EVIDENCE, exist_ok=True)

    if url is None:
        FAKE_CAMERA = FakeCamera().start()
        url = FAKE_URL
        print(f"Fake camera streaming at {url}")
    elif "TC08" in wanted:
        print("TC08 skipped with a real camera: turn the phone's Wi-Fi off and on by hand.")
        wanted.remove("TC08")

    results = []
    for tc in wanted:
        print(f"\n=== {tc} ===", flush=True)
        try:
            res = TESTS[tc](url)
        except Exception as exc:                      # a broken test is a failed test
            res = result(tc, tc, "", "", f"Test error: {exc}", False, [])
        results.append(res)
        print(f"{'PASS' if res['passed'] else 'FAIL'}  {res['id']} {res['name']}: {res['actual']}")

    if FAKE_CAMERA:
        FAKE_CAMERA.stop()
    summary = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "camera_url": url, "results": results}
    json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed. Details: test_output/results.json")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
