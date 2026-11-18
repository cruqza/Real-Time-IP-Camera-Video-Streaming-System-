import cv2
import os
import sys
import time
from datetime import datetime


def _close_cap(cap):
    """Release a VideoCapture if it is open. Safe to call with None."""
    if cap is not None:
        cap.release()


def _open_capture(url):
    """Open the HTTP stream with a short read timeout.
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    timeout_prop = getattr(cv2, "CAP_PROP_TIMEOUT", None)
    if timeout_prop is not None:
        cap.set(timeout_prop, 500)
    return cap


def _read_with_retry(cap, url, max_retries=5):
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
            return frame

        # read() returned (False, None): no frame this cycle.
        print("  No frame received. Reconnecting in 1s...")
        _close_cap(cap)
        time.sleep(1.0)
        cap = _open_capture(url)

    # Retries exhausted; the stream is genuinely dead.
    return None


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
        
        )
        sys.exit(1)
    
    # Capture + recording state.
    cap = None
    out = None
    recording = False
    frame_count = 0
    start_time = time.time()

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
            frame = _read_with_retry(cap, ipcam_url)
            if frame is None:
                # Stream stayed dead after several reconnection attempts.
                print("Could not recover the video stream. Exiting.")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            fps_est = frame_count / elapsed if elapsed > 0 else 0.0

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
                    out = _start_recording(frame)
                    recording = True

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


def _start_recording(frame):
    """Start recording the processed frame to a timestamped MP4."""
    path = f"recording_{_timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (frame.shape[1], frame.shape[0]), isColor=True)
    if not writer.isOpened():
        print(f"Failed to start recording -> {path}")
        return None
    print(f"Recording started -> {path}")
    return writer


def _stop_recording(writer):
    """Release a video writer."""
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()
