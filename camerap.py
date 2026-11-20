import cv2
import os
import sys
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


def _read_with_retry(cap, url, max_retries=5):
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
            print(f"  Stream error: {exc}. Reconnecting in 1s...")
            _close_cap(cap)
            time.sleep(1.0)
            cap = _open_capture(url)
            continue

        if ret:
            return frame, cap

        # read() returned (False, None): no frame this cycle.
        print("  No frame received. Reconnecting in 1s...")
        _close_cap(cap)
        time.sleep(1.0)
        cap = _open_capture(url)

    # Retries exhausted; the stream is genuinely dead.
    return None, cap


def _cleanup(cap, out):

    _close_cap(cap)
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def main():
    # Camera address / credentials come from the environment or a local
    # .env file -- never hardcoded in this repo (see .gitignore).
    ipcam_url = os.getenv("CAMERA_URL")

    if ipcam_url is None:
        _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        try:
            with open(_env_path) as _f:
                for _line in _f:
                    if _line.startswith("CAMERA_URL="):
                        ipcam_url = _line.split("=", 1)[1].strip()
                        break
        except OSError:
            pass

    if not ipcam_url:
        print(
            "Error: camera address not configured. Set CAMERA_URL in your "
            "environment or in a .env file next to this script, e.g.\n"
            "  CAMERA_URL=http://192.168.1.50:8080/video"
        )
        sys.exit(1)

    url_error = _validate_url(ipcam_url)
    if url_error:
        print(f"Error: {url_error}")
        sys.exit(1)

    # Capture + recording state.
    cap = None
    out = None
    recording = False
    rec_start = 0.0
    fps_meter = FpsMeter(window=1.0)
    message = ""          # short on-screen confirmation text
    message_until = 0.0   # time.time() after which it disappears

    try:
        cap = _open_capture(ipcam_url)
        if not cap.isOpened():
            # Wrong URL / camera off / wrong network -- clear message, clean exit.
            raise ConnectionError(
                f"Could not connect to the camera at {ipcam_url}. "
                "Check the URL, that the camera is powered on, "
                "and that it is on the same Wi-Fi."
            )

        print(f"Connected to camera at {ipcam_url}")

        while True:
            # Keep the returned cap: it is a new object after a reconnect.
            frame, cap = _read_with_retry(cap, ipcam_url)
            if frame is None:
                # Stream stayed dead after several reconnection attempts.
                print("Could not recover the video stream. Exiting.")
                break

            fps_meter.tick()
            fps_est = fps_meter.fps

            frame = process_frame(
                frame,
                camera_name="Phone IP Camera",
                fps_est=fps_est,
            )

            # Recording: write the processed frame (without the REC badge).
            if recording and out is not None:
                out.write(frame)

            # Status layer for the window only: REC badge and confirmations.
            display = frame.copy()
            if recording:
                _draw_rec_status(display, time.time() - rec_start)
            if message and time.time() < message_until:
                _draw_message(display, message)

            cv2.imshow("Phone IP Camera", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc -> exit
                break
            elif key in (ord("s"), ord("S")):  # S -> screenshot
                path = _capture_screenshot(frame)
                message = (f"Screenshot saved: {os.path.basename(path)}"
                           if path else "Screenshot failed")
                message_until = time.time() + MESSAGE_SECONDS
            elif key in (ord("r"), ord("R")):  # R -> toggle recording
                if recording:
                    _stop_recording(out)
                    out = None
                    recording = False
                    print("Recording stopped and saved.")
                    message = "Recording saved"
                else:
                    out = _start_recording(frame, fps_est)
                    # Only flag recording when the writer really opened.
                    recording = out is not None
                    rec_start = time.time()
                    message = "Recording started" if recording else "Recording failed"
                message_until = time.time() + MESSAGE_SECONDS

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if recording and out is not None:
            print("Recording finished and saved.")
        _cleanup(cap, out)


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


def _draw_rec_status(frame, elapsed):
    """Draw a red REC dot and the elapsed time in the top-right corner."""
    mins, secs = divmod(int(elapsed), 60)
    text = f"REC {mins:02d}:{secs:02d}"
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


def _capture_screenshot(frame):
    """Save the annotated current frame to screenshots/ as a timestamped JPG.

    Returns the saved path, or None on failure.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOT_DIR, f"screenshot_{_timestamp()}.jpg")
    if cv2.imwrite(path, frame):
        print(f"Screenshot saved -> {path}")
        return path
    print(f"Failed to save screenshot to {path}")
    return None


def _start_recording(frame, fps_est):
    """Start recording the processed frame to a timestamped MP4.

    The file's frame rate matches the stream's measured FPS, so playback runs
    at real speed. Returns None if the writer could not be opened.
    """
    fps = fps_est if fps_est >= MIN_RECORD_FPS else DEFAULT_RECORD_FPS
    fps = min(fps, MAX_RECORD_FPS)

    os.makedirs(RECORDING_DIR, exist_ok=True)
    path = os.path.join(RECORDING_DIR, f"recording_{_timestamp()}.mp4")
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