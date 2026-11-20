
import cv2
import os
import sys
import time
from collections import deque
from datetime import datetime

# Connection timeouts in milliseconds (needs OpenCV 4.5.2+).
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 2000

# Recording frame rate limits, used when the measured FPS is not usable yet.
DEFAULT_RECORD_FPS = 15.0
MIN_RECORD_FPS = 1.0
MAX_RECORD_FPS = 60.0


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
            "environment or in a .env file next to this script"
        )
        sys.exit(1)

    # Capture + recording state.
    cap = None
    out = None
    recording = False
    fps_meter = FpsMeter(window=1.0)

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

            # Recording: write the processed (resized) frame.
            if recording and out is not None:
                out.write(frame)

            cv2.imshow("Phone IP Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc -> exit
                break
            elif key == ord("s"):  # S -> screenshot
                _capture_screenshot(frame)
            elif key == ord("r"):  # R -> toggle recording
                if recording:
                    _stop_recording(out)
                    out = None
                    recording = False
                    print("Recording stopped.")
                else:
                    out = _start_recording(frame, fps_est)
                    # Only flag recording when the writer really opened.
                    recording = out is not None

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
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


def _capture_screenshot(frame):
    """Save the annotated current frame to a timestamped JPG."""
    path = f"screenshot_{_timestamp()}.jpg"
    if cv2.imwrite(path, frame):
        print(f"Screenshot saved -> {path}")
    else:
        print(f"Failed to save screenshot to {path}")


def _start_recording(frame, fps_est):
    """Start recording the processed frame to a timestamped MP4.

    The file's frame rate matches the stream's measured FPS, so playback runs
    at real speed. Returns None if the writer could not be opened.
    """
    fps = fps_est if fps_est >= MIN_RECORD_FPS else DEFAULT_RECORD_FPS
    fps = min(fps, MAX_RECORD_FPS)

    path = f"recording_{_timestamp()}.mp4"
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
