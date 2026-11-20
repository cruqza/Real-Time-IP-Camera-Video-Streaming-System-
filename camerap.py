import cv2
import math
import numpy as np
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlparse

# Connection timeouts in milliseconds (needs OpenCV 4.5.2+).
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 2000

# Recording frame rate limits, used when the measured FPS is not usable yet.
DEFAULT_RECORD_FPS = 15.0
MIN_RECORD_FPS = 1.0
MAX_RECORD_FPS = 60.0

# Output folders, created next to this script on first use.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
RECORDING_DIR = os.path.join(BASE_DIR, "recordings")

# How long an on-screen confirmation stays visible, in seconds.
MESSAGE_SECONDS = 2.0

VALID_SCHEMES = ("http", "https", "rtsp")

# Motion detection settings.
MOTION_WORK_WIDTH = 320      # frames are shrunk to this width before analysis (faster, less noise)
MOTION_MIN_AREA = 0.002      # ignore changed regions smaller than 0.2% of the frame
MOTION_WARMUP_FRAMES = 30    # frames used to learn the background before reporting motion

# Object detection (YOLO) settings.
YOLO_MODEL = "yolov8n.pt"    # nano model: fastest on CPU, downloaded automatically on first run
YOLO_CONFIDENCE = 0.40       # hide detections below 40% confidence
YOLO_IMAGE_SIZE = 640        # inference resolution
YOLO_EVERY_N_FRAMES = 3      # run YOLO on every 3rd frame and reuse boxes in between (keeps FPS up)

# Automatic recording on motion.
AUTO_RECORD_ENABLED = True       # start with auto-recording armed (toggle with A)
AUTO_RECORD_TRIGGER_FRAMES = 3   # consecutive frames with motion needed to start (ignores one-frame flicker)
AUTO_RECORD_QUIET_SECONDS = 5.0  # stop after this long with no motion
AUTO_RECORD_PREROLL_SECONDS = 2.0  # frames kept from just before the motion started

# Multiple cameras.
MAX_CAMERAS = 9              # number keys 1-9 select a camera
WINDOW_NAME = "IP Cameras"
SELECTED_COLOR = (0, 255, 255)  # yellow border around the selected camera
MAX_WINDOW_WIDTH = 1900      # the grid is scaled down to fit within this size
MAX_WINDOW_HEIGHT = 1000     # (portrait phone streams make tall tiles)


def _tile_width(count):
    """Width of each camera tile, so the whole grid fits on a normal screen."""
    if count == 1:
        return 960
    if count <= 4:
        return 640
    return 480


def _validate_url(url):
    """Return an error message if the URL is malformed, else None."""
    parsed = urlparse(url)
    if parsed.scheme not in VALID_SCHEMES:
        return (
            f"Invalid camera URL '{url}'. It must start with "
            "http://, https:// or rtsp://, e.g. http://192.168.1.50:8080/video"
        )
    if not parsed.hostname:
        return f"Invalid camera URL '{url}'. It has no host or IP address."
    try:
        parsed.port  # raises ValueError for a non-numeric or out-of-range port
    except ValueError:
        return f"Invalid camera URL '{url}'. The port is not a valid number."
    return None


def _close_cap(cap):
    """Release a VideoCapture if it is open. Safe to call with None."""
    if cap is not None:
        cap.release()


def _open_capture(url):
    """Open the stream with open and read timeouts.

    The timeouts must be passed when the capture is created; setting them
    afterwards with cap.set() has no effect. Older OpenCV builds without these
    properties fall back to a plain open.
    """
    open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)

    if open_prop is not None and read_prop is not None:
        params = [open_prop, OPEN_TIMEOUT_MS, read_prop, READ_TIMEOUT_MS]
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    else:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class MotionDetector:
    """Find the areas of a frame that changed, using background subtraction.

    OpenCV's MOG2 subtractor learns what the static scene looks like and marks
    pixels that differ from it. The mask is cleaned up (blur, threshold,
    morphology) and turned into bounding boxes around each moving region.
    """

    def __init__(self, work_width=MOTION_WORK_WIDTH, min_area=MOTION_MIN_AREA,
                 warmup_frames=MOTION_WARMUP_FRAMES):
        self.work_width = work_width
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.frames_seen = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=True
        )

    def reset(self):
        """Forget the learned background (e.g. after a reconnect or camera move)."""
        self.frames_seen = 0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=True
        )

    def detect(self, frame):
        """Return a list of (x, y, w, h) boxes in `frame` coordinates."""
        h, w = frame.shape[:2]
        scale = self.work_width / w
        small = cv2.resize(frame, (self.work_width, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        mask = self.subtractor.apply(small)
        self.frames_seen += 1
        if self.frames_seen < self.warmup_frames:
            return []

        # MOG2 marks shadows as 127; keep only real foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)            # remove speckles
        mask = cv2.dilate(mask, self.kernel, iterations=2)                    # join nearby blobs

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_pixels = self.min_area * mask.shape[0] * mask.shape[1]

        boxes = []
        for c in contours:
            if cv2.contourArea(c) < min_pixels:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Scale the box back up to the full frame size.
            boxes.append((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)))
        return boxes


def _draw_motion(frame, boxes):
    """Highlight moving regions: translucent red fill plus a red outline.

    `boxes` must already be in this frame's coordinates.
    """
    if not boxes:
        return
    overlay = frame.copy()
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)


def _draw_motion_status(frame, enabled, moving):
    """Show the motion detector state under the FPS line."""
    if not enabled:
        text, color = "Motion: OFF (M)", (160, 160, 160)
    elif moving:
        text, color = "MOTION DETECTED", (0, 0, 255)
    else:
        text, color = "Motion: ON (M)", (0, 255, 0)
    cv2.putText(frame, text, (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2 if moving else 1, cv2.LINE_AA)


class ObjectDetector:
    """Detect objects with a YOLO model from the `ultralytics` package.

    If the package or model can't be loaded, `available` is False and the
    viewer keeps running without object detection.
    """

    def __init__(self, model_path=YOLO_MODEL, confidence=YOLO_CONFIDENCE,
                 image_size=YOLO_IMAGE_SIZE):
        self.confidence = confidence
        self.image_size = image_size
        self.model = None
        self.error = None
        try:
            from ultralytics import YOLO  # imported here so the app runs without it
            self.model = YOLO(model_path)
        except ImportError:
            self.error = "ultralytics is not installed (pip install ultralytics)"
        except Exception as exc:  # bad path, failed download, etc.
            self.error = f"could not load YOLO model '{model_path}': {exc}"

    @property
    def available(self):
        return self.model is not None

    def detect(self, frame):
        """Return a list of (x1, y1, x2, y2, label, confidence) in `frame` coordinates."""
        result = self.model(frame, imgsz=self.image_size, conf=self.confidence,
                            verbose=False)[0]
        names = result.names
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append((x1, y1, x2, y2, names[cls_id], float(box.conf[0])))
        return detections


# Distinct bright BGR colours for object classes.
_PALETTE = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0), (0, 128, 255),
            (255, 255, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128), (128, 255, 0)]


def _class_color(label):
    """A stable colour per class name, so each class keeps its colour."""
    return _PALETTE[sum(ord(c) for c in label) % len(_PALETTE)]


def _draw_detections(frame, detections):
    """Draw a box and a 'label 0.87' tag for every detected object."""
    for (x1, y1, x2, y2, label, conf) in detections:
        color = _class_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1, th + base + 4)  # keep the tag on screen at the top edge
        cv2.rectangle(frame, (x1, ty - th - base - 4), (x1 + tw + 6, ty), color, -1)
        cv2.putText(frame, tag, (x1 + 3, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_yolo_status(frame, available, enabled, count):
    """Show the object detector state under the motion line."""
    if not available:
        text, color = "Objects: unavailable", (160, 160, 160)
    elif not enabled:
        text, color = "Objects: OFF (Y)", (160, 160, 160)
    else:
        text, color = f"Objects: {count} (Y)", (0, 255, 255)
    cv2.putText(frame, text, (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


class AutoRecorder:
    """Decide when motion should start or stop an automatic recording.

    Call update() once per frame with whether motion was seen. It returns
    "start", "stop" or None. It holds no video itself, so main() stays in
    charge of the writer.
    """

    def __init__(self, trigger_frames=AUTO_RECORD_TRIGGER_FRAMES,
                 quiet_seconds=AUTO_RECORD_QUIET_SECONDS):
        self.trigger_frames = trigger_frames
        self.quiet_seconds = quiet_seconds
        self.active = False          # an auto recording is running
        self.motion_streak = 0       # consecutive frames with motion
        self.last_motion = 0.0       # time motion was last seen

    def update(self, moving, now=None):
        now = time.time() if now is None else now
        if moving:
            self.motion_streak += 1
            self.last_motion = now
        else:
            self.motion_streak = 0

        if not self.active and self.motion_streak >= self.trigger_frames:
            self.active = True
            return "start"
        if self.active and now - self.last_motion >= self.quiet_seconds:
            self.active = False
            return "stop"
        return None

    def seconds_left(self, now=None):
        """Seconds of quiet left before an active recording stops."""
        now = time.time() if now is None else now
        return max(0.0, self.quiet_seconds - (now - self.last_motion))

    def cancel(self):
        """Forget the current auto recording (it was stopped some other way)."""
        self.active = False
        self.motion_streak = 0


def _draw_auto_status(frame, enabled, auto, quiet):
    """Show the auto-record state under the objects line."""
    if not enabled:
        text, color = "Auto-rec: OFF (A)", (160, 160, 160)
    elif auto.active and quiet:
        text, color = f"Auto-rec: stops in {auto.seconds_left():.0f}s", (0, 165, 255)
    elif auto.active:
        text, color = "Auto-rec: RECORDING", (0, 0, 255)
    else:
        text, color = "Auto-rec: armed (A)", (0, 255, 0)
    cv2.putText(frame, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


class FpsMeter:
    """Rolling FPS over the last `window` seconds, so drops show up quickly."""

    def __init__(self, window=1.0):
        self.window = window
        self.times = deque()

    def tick(self):
        now = time.time()
        self.times.append(now)
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


def _read_with_retry(cap, url, max_retries=5, name=""):
    """Read one frame, reconnecting on failure.

    Returns a (frame, cap) tuple. The capture may have been replaced by a
    fresh one during reconnection, so the caller must keep the returned cap.
    frame is None if every retry failed.
    """
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        try:
            ret, frame = cap.read()
        except (cv2.error, OSError) as exc:
            # Network-level failure: the stream dropped or the camera vanished.
            print(f"  [{name}] Stream error: {exc}. Reconnecting in 1s...")
            _close_cap(cap)
            time.sleep(1.0)
            cap = _open_capture(url)
            continue

        if ret:
            return frame, cap

        # read() returned (False, None): no frame this cycle.
        print(f"  [{name}] No frame received. Reconnecting in 1s...")
        _close_cap(cap)
        time.sleep(1.0)
        cap = _open_capture(url)

    # Retries exhausted; the stream is genuinely dead.
    return None, cap


def _read_env_file():
    """Read KEY=value lines from the .env file next to this script."""
    values = {}
    try:
        with open(os.path.join(BASE_DIR, ".env")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def _load_cameras():
    """Return a list of (name, url) for every configured camera.

    Accepted forms, in the environment or the .env file (environment wins):
      CAMERA_URLS=Front=http://..../video,Back=http://..../video   (names optional)
      CAMERA_URL_1=..., CAMERA_URL_2=...   with optional CAMERA_NAME_1=...
      CAMERA_URL=...                       one camera, as before
    """
    cfg = _read_env_file()
    cfg.update({k: v for k, v in os.environ.items() if k.startswith("CAMERA_")})

    cameras = []
    if cfg.get("CAMERA_URLS"):
        items = [i.strip() for i in cfg["CAMERA_URLS"].split(",") if i.strip()]
        for n, item in enumerate(items, start=1):
            # "Name=url" only when the '=' comes before the '://'
            if "=" in item and "://" in item and item.index("=") < item.index("://"):
                name, url = item.split("=", 1)
            else:
                name, url = f"Camera {n}", item
            cameras.append((name.strip(), url.strip()))
    else:
        n = 1
        while cfg.get(f"CAMERA_URL_{n}"):
            cameras.append((cfg.get(f"CAMERA_NAME_{n}", f"Camera {n}"), cfg[f"CAMERA_URL_{n}"]))
            n += 1
        if not cameras and cfg.get("CAMERA_URL"):
            cameras.append(("Phone IP Camera", cfg["CAMERA_URL"]))

    if len(cameras) > MAX_CAMERAS:
        print(f"Only the first {MAX_CAMERAS} cameras are used.")
    return cameras[:MAX_CAMERAS]


class CameraStream:
    """Read one camera in a background thread.

    Each camera gets its own thread, so a slow or reconnecting stream never
    freezes the others. The main loop just picks up the newest frame.
    """

    def __init__(self, name, url, max_retries=5):
        self.name = name
        self.url = url
        self.max_retries = max_retries
        self.status = "connecting"   # connecting | live | lost
        self.error = None
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        cap = _open_capture(self.url)
        if not cap.isOpened():
            _close_cap(cap)
            self.error = ("could not connect. Check the URL, that the camera is on, "
                          "and that it is on the same Wi-Fi")
            self.status = "lost"
            return
        self.status = "live"
        while not self._stop.is_set():
            frame, cap = _read_with_retry(cap, self.url, self.max_retries, self.name)
            if frame is None:
                self.error = "stream lost after several reconnection attempts"
                self.status = "lost"
                break
            with self._lock:
                self._frame = frame
                self._frame_id += 1
        _close_cap(cap)

    def read(self):
        """Return (newest frame or None, frame counter)."""
        with self._lock:
            return self._frame, self._frame_id

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


class Camera:
    """One camera's stream plus its own detectors, recorder and overlays."""

    def __init__(self, number, name, url, tile_width, yolo_available):
        self.number = number                     # 1-based, matches the number key
        self.name = name
        self.label = f"[{number}] {name}"
        self.slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or f"cam{number}"
        self.tile_width = tile_width
        self.stream = CameraStream(name, url)

        self.fps = FpsMeter(window=1.0)
        self.motion = MotionDetector()
        self.auto = AutoRecorder()
        self.preroll = deque()

        self.motion_enabled = True
        self.yolo_enabled = yolo_available
        self.auto_enabled = AUTO_RECORD_ENABLED

        self.out = None
        self.recording = False
        self.rec_mode = None                     # "manual" or "auto"
        self.rec_start = 0.0

        self.boxes = []
        self.detections = []
        self.frame_index = 0
        self.last_id = 0
        self.tile = None                         # latest processed frame
        self.yolo_error = None
        self.lost_reported = False

    # -- recording -------------------------------------------------------
    def start_recording(self, mode):
        if self.tile is None:
            return False
        prefix = "motion" if mode == "auto" else "recording"
        self.out = _start_recording(self.tile, self.fps.fps, prefix=f"{prefix}_{self.slug}")
        if self.out is None:
            return False
        self.recording, self.rec_mode, self.rec_start = True, mode, time.time()
        if mode == "manual":
            self.preroll.clear()
        return True

    def stop_recording(self):
        _stop_recording(self.out)
        self.out, self.recording, self.rec_mode = None, False, None
        self.auto.cancel()
        print(f"[{self.name}] Recording stopped and saved.")

    def screenshot(self):
        return _capture_screenshot(self.tile, tag=self.slug) if self.tile is not None else None

    # -- per-frame processing -------------------------------------------
    def update(self, yolo):
        """Process the newest frame, if a new one arrived. Returns a message or None."""
        raw, frame_id = self.stream.read()
        if raw is None or frame_id == self.last_id:
            return None
        self.last_id = frame_id
        message = None

        self.fps.tick()
        raw_w = raw.shape[1]

        # Detectors run on the clean raw frame, before any text is drawn.
        self.boxes = self.motion.detect(raw) if self.motion_enabled else []

        self.frame_index += 1
        if not self.yolo_enabled:
            self.detections = []
        elif self.frame_index % YOLO_EVERY_N_FRAMES == 0:
            try:
                self.detections = yolo.detect(raw)
            except Exception as exc:
                self.yolo_error = exc
                self.yolo_enabled = False
                self.detections = []

        frame = process_frame(raw, camera_name=self.label, fps_est=self.fps.fps,
                              display_width=self.tile_width)
        k = frame.shape[1] / raw_w
        if self.boxes:
            _draw_motion(frame, [(int(x * k), int(y * k), int(w * k), int(h * k))
                                 for (x, y, w, h) in self.boxes])
        if self.detections:
            _draw_detections(frame, [
                (int(x1 * k), int(y1 * k), int(x2 * k), int(y2 * k), label, conf)
                for (x1, y1, x2, y2, label, conf) in self.detections
            ])
        self.tile = frame

        # Automatic recording on motion.
        now = time.time()
        if self.auto_enabled and self.motion_enabled and self.rec_mode != "manual":
            action = self.auto.update(bool(self.boxes), now)
            if action == "start":
                if self.start_recording("auto"):
                    for _, old in self.preroll:
                        if old.shape == frame.shape:
                            self.out.write(old)
                    self.preroll.clear()
                    message = f"{self.name}: auto-recording started"
                else:
                    self.auto.cancel()
                    message = f"{self.name}: auto-recording failed"
            elif action == "stop" and self.rec_mode == "auto":
                self.stop_recording()
                message = f"{self.name}: auto-recording saved"

        if self.auto_enabled and not self.recording:
            self.preroll.append((now, frame.copy()))
            while self.preroll and now - self.preroll[0][0] > AUTO_RECORD_PREROLL_SECONDS:
                self.preroll.popleft()

        if self.recording and self.out is not None:
            self.out.write(frame)
        return message

    # -- window-only rendering ------------------------------------------
    def render(self, yolo_available, selected):
        """The tile as shown in the window: frame plus status lines and badges."""
        if self.tile is None:
            h = self.tile_width * 9 // 16
            display = np.zeros((h, self.tile_width, 3), np.uint8)
            if self.stream.status == "lost":
                text, color = "Could not connect", (0, 0, 255)
            else:
                text, color = "Connecting...", (200, 200, 200)
            cv2.putText(display, self.label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2, cv2.LINE_AA)
        else:
            display = self.tile.copy()
            _draw_motion_status(display, self.motion_enabled, bool(self.boxes))
            _draw_yolo_status(display, yolo_available, self.yolo_enabled,
                              len(self.detections) if self.yolo_enabled else 0)
            _draw_auto_status(display, self.auto_enabled, self.auto, not self.boxes)
            if self.recording:
                _draw_rec_status(display, time.time() - self.rec_start,
                                 "AUTO REC" if self.rec_mode == "auto" else "REC")
            if self.stream.status == "lost":
                h, w = display.shape[:2]
                cv2.rectangle(display, (0, h // 2 - 25), (w, h // 2 + 15), (0, 0, 0), -1)
                cv2.putText(display, "SIGNAL LOST", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if selected:
            h, w = display.shape[:2]
            cv2.rectangle(display, (1, 1), (w - 2, h - 2), SELECTED_COLOR, 3)
        return display

    def close(self):
        if self.recording:
            _stop_recording(self.out)
            self.out, self.recording = None, False
            print(f"[{self.name}] Recording finished and saved.")
        self.stream.stop()


def _make_grid(tiles):
    """Arrange equal-width tiles in a near-square grid, padding heights with black."""
    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = max(t.shape[1] for t in tiles)
    cell_h = max(t.shape[0] for t in tiles)
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * cell_h:r * cell_h + t.shape[0], c * cell_w:c * cell_w + t.shape[1]] = t

    # Shrink the whole grid if it would not fit on screen (recordings are unaffected).
    scale = min(1.0, MAX_WINDOW_WIDTH / grid.shape[1], MAX_WINDOW_HEIGHT / grid.shape[0])
    if scale < 1.0:
        grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
    return grid


def main():
    # Camera addresses / credentials come from the environment or a local
    # .env file -- never hardcoded in this repo (see .gitignore).
    configured = _load_cameras()
    if not configured:
        print(
            "Error: camera address not configured. Set CAMERA_URL in your "
            "environment or in a .env file next to this script, e.g.\n"
            "  CAMERA_URL=http://192.168.1.50:8080/video\n"
            "For several cameras use CAMERA_URLS (comma-separated) or "
            "CAMERA_URL_1, CAMERA_URL_2, ..."
        )
        sys.exit(1)

    for name, url in configured:
        url_error = _validate_url(url)
        if url_error:
            print(f"Error ({name}): {url_error}")
            sys.exit(1)

    print(f"Loading YOLO model {YOLO_MODEL}...")
    yolo = ObjectDetector()
    if yolo.available:
        print("YOLO ready.")
    else:
        print(f"Object detection disabled: {yolo.error}")

    width = _tile_width(len(configured))
    cameras = [Camera(i, name, url, width, yolo.available)
               for i, (name, url) in enumerate(configured, start=1)]

    selected = 0          # 0 = all cameras, 1-9 = one camera
    message = ""          # short on-screen confirmation text
    message_until = 0.0   # time.time() after which it disappears

    def say(text):
        nonlocal message, message_until
        message, message_until = text, time.time() + MESSAGE_SECONDS
        print(text)

    try:
        for cam in cameras:
            print(f"Connecting to {cam.name} at {cam.stream.url}...")
            cam.stream.start()

        while True:
            # Process the newest frame from each camera.
            for cam in cameras:
                note = cam.update(yolo)
                if note:
                    say(note)
                if cam.yolo_error is not None:
                    say(f"{cam.name}: YOLO error, object detection turned off")
                    cam.yolo_error = None
                if cam.stream.status == "lost" and not cam.lost_reported:
                    cam.lost_reported = True
                    print(f"[{cam.name}] {cam.stream.error}.")
                    if cam.recording:
                        cam.stop_recording()

            if all(c.stream.status == "lost" for c in cameras):
                if any(c.tile is not None for c in cameras):
                    print("Could not recover any video stream. Exiting.")
                else:
                    print("Could not connect to any camera. Check the URLs, that the "
                          "cameras are powered on, and that they are on the same Wi-Fi.")
                break

            # Build the window: every tile, the selection border, then messages.
            tiles = [c.render(yolo.available, selected == c.number) for c in cameras]
            grid = _make_grid(tiles)
            if message and time.time() < message_until:
                _draw_message(grid, message)
            cv2.imshow(WINDOW_NAME, grid)

            key = cv2.waitKey(5) & 0xFF
            if key == 255:
                continue
            targets = cameras if selected == 0 else [cameras[selected - 1]]
            live = [c for c in targets if c.tile is not None]

            if key == 27:  # Esc -> exit
                break
            elif ord("0") <= key <= ord("9"):  # 0 = all, 1-9 = one camera
                n = key - ord("0")
                if n == 0 or n <= len(cameras):
                    selected = n
                    say("Controls: all cameras" if n == 0
                        else f"Controls: {cameras[n - 1].label}")
            elif key in (ord("s"), ord("S")):  # S -> screenshot
                saved = [p for p in (c.screenshot() for c in live) if p]
                say(f"Screenshot saved: {os.path.basename(saved[0])}" if len(saved) == 1
                    else f"{len(saved)} screenshots saved" if saved else "Screenshot failed")
            elif key in (ord("r"), ord("R")):  # R -> toggle recording
                if any(c.recording for c in live):
                    for c in live:
                        if c.recording:
                            c.stop_recording()
                    say("Recording saved")
                else:
                    ok = [c for c in live if c.start_recording("manual")]
                    say(f"Recording started ({len(ok)} camera{'s' if len(ok) != 1 else ''})"
                        if ok else "Recording failed")
            elif key in (ord("m"), ord("M")):  # M -> toggle motion detection
                on = not all(c.motion_enabled for c in targets)
                for c in targets:
                    c.motion_enabled = on
                    c.motion.reset()  # relearn the background when switched back on
                say(f"Motion detection {'ON' if on else 'OFF'}")
            elif key in (ord("y"), ord("Y")):  # Y -> toggle object detection
                if yolo.available:
                    on = not all(c.yolo_enabled for c in targets)
                    for c in targets:
                        c.yolo_enabled = on
                        c.detections = []
                    say(f"Object detection {'ON' if on else 'OFF'}")
                else:
                    say("Object detection unavailable: pip install ultralytics")
            elif key in (ord("a"), ord("A")):  # A -> arm/disarm auto-recording
                on = not all(c.auto_enabled for c in targets)
                for c in targets:
                    if not on and c.rec_mode == "auto":
                        c.stop_recording()
                    c.auto_enabled = on
                    c.auto.cancel()
                    c.preroll.clear()
                text = f"Auto-recording {'armed' if on else 'OFF'}"
                if on and not all(c.motion_enabled for c in targets):
                    text += " (turn motion on with M)"
                say(text)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        # Safe shutdown: finish recordings, stop every stream, close windows.
        for cam in cameras if "cameras" in locals() else []:
            cam.close()
        cv2.destroyAllWindows()


def process_frame(frame, camera_name, fps_est, display_width=960):
    """Resize a frame to a fixed display width, keeping aspect ratio, then overlay
    the current date/time, camera name, resolution and estimated FPS."""
    height, width = frame.shape[:2]
    display_height = int(height * display_width / width)
    frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resolution = f"{display_width}x{display_height}"

    lines = [
        camera_name,
        f"{resolution} | {now}",
        f"FPS: {fps_est:.1f}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return frame


def _timestamp():
    """Return a YYYYMMDD_HHMMSS string for use in file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _draw_rec_status(frame, elapsed, label="REC"):
    """Draw a red REC dot and the elapsed time in the top-right corner."""
    mins, secs = divmod(int(elapsed), 60)
    text = f"{label} {mins:02d}:{secs:02d}"
    w = frame.shape[1]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = w - tw - 20
    # Blink the dot once a second.
    if int(elapsed * 2) % 2 == 0:
        cv2.circle(frame, (x - 14, 26), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA)


def _draw_message(frame, text):
    """Draw a short confirmation at the bottom of the frame on a dark band."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def _capture_screenshot(frame, tag=""):
    """Save the annotated current frame to screenshots/ as a timestamped JPG.

    `tag` (the camera's short name) goes in the file name. Returns the saved
    path, or None on failure.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    name = f"screenshot_{tag}_{_timestamp()}.jpg" if tag else f"screenshot_{_timestamp()}.jpg"
    path = os.path.join(SCREENSHOT_DIR, name)
    if cv2.imwrite(path, frame):
        print(f"Screenshot saved -> {path}")
        return path
    print(f"Failed to save screenshot to {path}")
    return None


def _start_recording(frame, fps_est, prefix="recording"):
    """Start recording the processed frame to a timestamped MP4.

    The file's frame rate matches the stream's measured FPS, so playback runs
    at real speed. Returns None if the writer could not be opened.
    """
    fps = fps_est if fps_est >= MIN_RECORD_FPS else DEFAULT_RECORD_FPS
    fps = min(fps, MAX_RECORD_FPS)

    os.makedirs(RECORDING_DIR, exist_ok=True)
    # Manual clips are recording_*.mp4, motion clips are motion_*.mp4.
    path = os.path.join(RECORDING_DIR, f"{prefix}_{_timestamp()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (frame.shape[1], frame.shape[0]), isColor=True)
    if not writer.isOpened():
        print(f"Failed to start recording -> {path}")
        return None
    print(f"Recording started at {fps:.1f} FPS -> {path}")
    return writer


def _stop_recording(writer):
    """Release a video writer."""
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()