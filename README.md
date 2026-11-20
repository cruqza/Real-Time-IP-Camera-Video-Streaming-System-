# Phone IP Camera Viewer

Shows one camera, or up to nine side by side in a grid.

A small Python + OpenCV app that streams live video from a phone (or any IP camera) over your local Wi-Fi, shows it in a desktop window with an on-screen info overlay, and lets you take screenshots and record video with single key presses.

## What it does

1. **Finds the camera address** from the `CAMERA_URL` environment variable, or from a `.env` file next to the script. The address is never hardcoded, so credentials stay out of the repo.
2. **Connects to the stream** with OpenCV's FFmpeg backend, using a one-frame buffer to keep latency low. It gives up opening after 5 seconds, and treats a read as failed after 2 seconds with no frame.
3. **Shows each frame** in a window called "Phone IP Camera", after:
   - resizing it to 960 px wide (keeping the aspect ratio)
   - drawing a green overlay with the camera name, resolution, current date and time, and a rolling FPS (frames over the last second)
4. **Reconnects automatically** if the stream drops. It retries up to 5 times, 1 second apart, before giving up. After a successful reconnect the main loop carries on with the new connection.
5. **Handles keyboard controls** for screenshots, recording and exit.
6. **Cleans up** on exit, Ctrl+C or an error: releases the camera, closes any recording file and shuts the window.

## Requirements

- Python 3.7+
- OpenCV 4.5.2 or newer with FFmpeg support (older versions still run, but without the connection timeouts):

```bash
pip install opencv-python
pip install ultralytics   # optional, for YOLO object detection (also installs PyTorch)
```

- A phone running an IP camera app (for example "IP Webcam" on Android) or any camera that serves an HTTP/RTSP video stream
- The computer and the camera on the **same Wi-Fi network**

## Setup

Set the camera address in one of two ways.

**Option 1: environment variable**

```bash
# macOS / Linux
export CAMERA_URL="http://192.168.1.50:8080/video"

# Windows (PowerShell)
$env:CAMERA_URL = "http://192.168.1.50:8080/video"
```

**Option 2: `.env` file** in the same folder as the script:

```
CAMERA_URL=http://192.168.1.50:8080/video
```

Add `.env` to `.gitignore` so the address and any credentials are never committed.

## Usage

```bash
python camerap.py
```

(Use whatever name you gave the script.)

### Keyboard controls

The video window must be focused.

| Key   | Action                                                              |
|-------|---------------------------------------------------------------------|
| `S`   | Save a screenshot to `screenshots/screenshot_YYYYMMDD_HHMMSS.jpg`, with a 2-second on-screen confirmation |
| `R`   | Start or stop recording to `recordings/recording_YYYYMMDD_HHMMSS.mp4`, at the stream's measured FPS. A blinking red REC badge and the elapsed time show while it records |
| `M`   | Turn motion detection on or off (on at startup)                     |
| `Y`   | Turn YOLO object detection on or off (on at startup if available)  |
| `A`   | Arm or disarm automatic recording on motion (armed at startup)     |
| `Esc` | Quit                                                                |

Screenshots go to `screenshots/` and recordings to `recordings/`, both next to the script and created if they're missing. Both include the info overlay. The REC badge and confirmations show only in the window.

## Motion detection

The viewer highlights the parts of the picture that are moving. Each moving region gets a translucent red fill and a red box, and the status line under the FPS reads **MOTION DETECTED** while anything moves. Otherwise it shows `Motion: ON (M)`, or `Motion: OFF (M)` when you've switched it off.

**How it works** (`MotionDetector` class):

1. Each raw frame is shrunk to 320 px wide and blurred, which is faster and ignores sensor noise.
2. OpenCV's **MOG2 background subtractor** learns what the still scene looks like and marks the pixels that differ from it. It adapts slowly, so changes in lighting fade into the background.
3. The mask is cleaned up. Shadows (value 127) are dropped by a threshold, an *open* removes speckles, and a *dilate* joins nearby blobs.
4. Contours smaller than 0.2% of the frame are ignored. The rest become bounding boxes, which are scaled back up to the display frame.
5. For the first 30 frames nothing is reported, while the background is being learned.

Detection runs on the clean frame **before** the text overlay is drawn, so the ticking clock never counts as motion. The highlights do appear in screenshots and recordings.

**Tuning** (constants at the top of `camerap.py`):

| Constant | Default | Effect |
|----------|---------|--------|
| `MOTION_WORK_WIDTH` | 320 | Analysis resolution. Lower is faster but less precise |
| `MOTION_MIN_AREA` | 0.002 | Smallest region reported, as a fraction of the frame. Raise it if leaves or noise trigger it |
| `MOTION_WARMUP_FRAMES` | 30 | Frames spent learning the background |

Pressing `M` also resets the learned background. That's useful if the phone was moved.

Tip: keep the phone still. If the camera itself moves, the whole frame counts as motion.

## Multiple cameras

The viewer can show **up to 9 streams at once** in a grid: 1 camera is 960 px wide, 2 to 4 are 640 px tiles in a 2x2 grid, and 5 to 9 are 480 px tiles in a 3x3 grid. If the grid is bigger than 1900x1000 it's scaled down to fit. Each tile has its own overlay, motion detection, YOLO boxes, auto-recording and REC badge.

### Configure the cameras

Use any one of these, in `.env` or as environment variables:

```
# Option A: one line, names optional
CAMERA_URLS=Front Door=http://192.168.1.50:8080/video,Back=http://192.168.1.51:8080/video

# Option B: numbered
CAMERA_URL_1=http://192.168.1.50:8080/video
CAMERA_NAME_1=Front Door
CAMERA_URL_2=http://192.168.1.51:8080/video
CAMERA_NAME_2=Back

# Option C: one camera, as before
CAMERA_URL=http://192.168.1.50:8080/video
```

A camera without a name is called "Camera N". Every URL is checked before anything starts.

### Choosing which camera the keys control

| Key | Action |
|-----|--------|
| `0` | Keys control **all** cameras (the default) |
| `1`-`9` | Keys control only that camera, marked with a yellow border |

`S`, `R`, `M`, `Y` and `A` then act on the selected camera, or on all of them. With all selected, `R` starts recording on every camera, or stops all of them if any is recording. `M`, `Y` and `A` turn the feature on for every camera unless it's already on for all of them.

### How it works

- **`CameraStream`**: each camera is read in its own background thread, so a slow or reconnecting camera never freezes the others. The main loop always takes the newest frame.
- **`Camera`**: holds everything per camera: its stream, FPS meter, motion detector, auto-recorder, pre-roll buffer, writer and on/off switches. `update()` processes a new frame, and `render()` adds the status lines, REC badge, selection border, and the "Connecting..." / "Could not connect" / "SIGNAL LOST" states.
- **One YOLO model** is shared by all cameras, so it only loads once.
- **`_make_grid()`** pads the tiles to equal height (portrait and landscape streams can mix) and tiles them in a near-square grid.

### Files

Names include the camera so files from different cameras never clash: `screenshots/screenshot_front_door_*.jpg`, `recordings/recording_front_door_*.mp4` (manual) and `recordings/motion_front_door_*.mp4` (auto).

### When a camera fails

If one camera can't connect or drops for good, its tile shows the problem, any recording on it is saved, and the others keep running. The app only exits when **every** camera is lost.

Tip: each stream with YOLO on adds CPU load. With 3 or more cameras, consider pressing `Y` for only the camera that needs it, or raise `YOLO_EVERY_N_FRAMES`.

## Automatic recording on motion

While auto-recording is armed (the default), the viewer records by itself whenever motion is detected. It stops once the scene has been quiet for 5 seconds.

**How it works** (`AutoRecorder` class plus the loop in `main()`):

1. Each frame, the motion detector's result (`boxes` not empty) goes to `AutoRecorder.update()`.
2. A recording **starts** after 3 frames in a row with motion, so a single flicker doesn't trigger it. The clip is saved as `recordings/motion_YYYYMMDD_HHMMSS.mp4`.
3. **Pre-roll**: while armed, the last 2 seconds of frames are kept in memory and written at the start of the clip, so you see what led up to the motion.
4. Each frame with motion resets the quiet timer. After 5 seconds with no motion the recording **stops** and the file is saved.
5. The window shows `AUTO REC mm:ss` in the top-right corner. The status line reads `Auto-rec: armed`, `RECORDING`, `stops in Ns` or `OFF`.

**How it works with the other controls:**

- Motion detection must be on (`M`). With motion off, nothing triggers.
- A manual recording (`R`) is never stopped by the auto timer, and auto won't start a second clip while one is running.
- Pressing `R` during an auto recording stops it straight away.
- Disarming with `A` stops and saves any auto clip in progress.
- Esc, Ctrl+C or an error still save the clip safely.

**Tuning** (constants at the top of `camerap.py`):

| Constant | Default | Effect |
|----------|---------|--------|
| `AUTO_RECORD_ENABLED` | True | Whether auto-recording is armed at startup |
| `AUTO_RECORD_TRIGGER_FRAMES` | 3 | Consecutive motion frames needed to start. Raise it if you get false starts |
| `AUTO_RECORD_QUIET_SECONDS` | 5.0 | Seconds of no motion before stopping |
| `AUTO_RECORD_PREROLL_SECONDS` | 2.0 | Seconds kept from before the motion began |

## Object detection (YOLO)

The viewer recognises everyday objects (people, phones, cups, laptops, cars and the rest of the 80 COCO classes) and draws a coloured box around each one, tagged with a label and confidence, such as `person 0.87`. Each class keeps its own colour. A status line reads `Objects: N (Y)`.

**How it works** (`ObjectDetector` class):

1. At startup the app loads **YOLOv8n** through the `ultralytics` package. The 6 MB model downloads automatically the first time, so you need internet then.
2. YOLO runs on the clean raw frame (before any overlay text) at 640 px, keeping detections at 40% confidence or higher.
3. To keep the stream smooth on a CPU, YOLO runs on **every 3rd frame**, and the last boxes are reused in between.
4. The boxes are scaled to the 960 px display frame and drawn by `_draw_detections`.

**If it's unavailable**: when `ultralytics` isn't installed or the model can't load, the app prints why and keeps running without object detection. The status line then shows `Objects: unavailable`. If inference fails partway through a session, object detection switches off with a message rather than crashing.

**Tuning** (constants at the top of `camerap.py`):

| Constant | Default | Effect |
|----------|---------|--------|
| `YOLO_MODEL` | `yolov8n.pt` | Swap in `yolov8s.pt` for better accuracy at a lower FPS |
| `YOLO_CONFIDENCE` | 0.40 | Raise it to cut false detections |
| `YOLO_IMAGE_SIZE` | 640 | Lower it (e.g. 416) for speed |
| `YOLO_EVERY_N_FRAMES` | 3 | Raise it if FPS drops; set it to 1 to run on every frame |

The boxes appear in screenshots and recordings. The status lines show only in the window.

## Browser dashboard (`dashboard.py`)

A Flask web dashboard that shows every camera live in the browser, with the same controls as the keyboard.

### Run it

```bash
pip install flask
python dashboard.py
```

Then open **http://localhost:5000**. From a phone or another PC on the same Wi-Fi, use `http://<this-PC's-IP>:5000`. It uses the same `.env` camera settings as `camerap.py`, and both files must be in the same folder.

### What's on the page

- **A live tile per camera**, the same frame as the desktop window: overlay, motion highlights, YOLO boxes, REC badge, and the "Could not connect" / "SIGNAL LOST" states
- **A status strip per camera**: live/connecting/lost, FPS, motion, object count, and a red `● REC mm:ss` while recording
- **Buttons per camera**: 📸 Screenshot, ⏺ Record / ⏹ Stop, Motion, YOLO, Auto-rec. Buttons that are on are highlighted
- **"All cameras" buttons** in the header
- **An Activity log** of recent events, such as auto-recordings starting and stopping, or a camera dropping
- **Saved files**: the newest 30 recordings and screenshots, which you can click to download

### How it works

| Part | What it does |
|------|--------------|
| `Dashboard` class | Owns the cameras (the same `Camera` objects as `camerap.py`) and one processing thread that updates every camera and keeps its latest frame as a JPEG |
| `GET /` | The dashboard page (HTML, CSS and JS in `PAGE`) |
| `GET /video/<n>` | Camera *n* as an MJPEG stream (`multipart/x-mixed-replace`), which shows in a plain `<img>` tag |
| `GET /api/status` | JSON with each camera's state and the recent activity. The page polls it every second |
| `POST /api/control` | `{"camera": 0-9, "action": "screenshot" \| "record" \| "motion" \| "yolo" \| "auto"}`. Camera `0` means all |
| `GET /api/files` | Lists saved recordings and screenshots |
| `GET /files/<kind>/<name>` | Downloads a file. Paths that try to leave the folder are refused |

All processing happens once, on the server, however many browsers are open. A lock keeps button presses from clashing with frame processing.

### Settings

| Variable / constant | Default | Effect |
|---------------------|---------|--------|
| `DASHBOARD_HOST` (env) | `0.0.0.0` | Set it to `127.0.0.1` to allow only this PC |
| `DASHBOARD_PORT` (env) | `5000` | Change it if the port is in use |
| `STREAM_FPS` | 15 | Maximum frames per second sent to each browser |
| `JPEG_QUALITY` | 80 | Lower it to save bandwidth |

Ctrl+C stops the server, saves any running recordings and releases the cameras.

**Note**: recordings use the `mp4v` codec, so download them and play them in VLC. Most browsers won't play them inline. The dashboard has no login, so only run it on a network you trust.

## Automated tests (`test.py`)

Runs test cases TC01-TC10 against `camerap.py` and saves evidence.

```bash
python test.py                 # all ten tests, using a built-in fake camera
python test.py TC06 TC07       # only some tests
python test.py --camera-url http://192.168.1.23:8080/video   # your real phone (TC08 skipped)
```

- A fake IP camera (MJPEG over HTTP, 1280x720, ~15 FPS) runs inside the script on port 8090 and can be switched off to simulate a Wi-Fi drop.
- Each test runs the real `camerap.main()` in its own process. Only `cv2.imshow`, `cv2.waitKey` and `cv2.destroyAllWindows` are replaced, to save the window's frames and press scripted keys. TC10 sends a real SIGINT (Ctrl+C).
- Output in `test_output/`: `evidence/` images, the screenshots and recordings each test saved, and `results.json`. Exit code 0 means every test passed.

## How the code is organised

| Function | Purpose |
|----------|---------|
| `main()` | Loads the cameras, starts their streams, builds the grid each frame and handles keys |
| `_open_capture(url)` | Opens the stream with a small buffer and open/read timeouts |
| `MotionDetector` | Background subtraction (MOG2) that returns boxes around moving regions |
| `_draw_motion(frame, boxes)` | Draws the red highlight boxes |
| `_draw_motion_status(...)` | Draws the motion ON/OFF/DETECTED status line |
| `ObjectDetector` | Loads YOLO and returns `(x1, y1, x2, y2, label, confidence)` per object |
| `_draw_detections(frame, detections)` | Draws the coloured boxes and labels |
| `_draw_yolo_status(...)` | Draws the Objects status line |
| `AutoRecorder` | Decides when motion starts or stops an automatic recording |
| `_draw_auto_status(...)` | Draws the Auto-rec status line |
| `_load_cameras()` | Reads the camera list from `CAMERA_URLS`, `CAMERA_URL_N` or `CAMERA_URL` |
| `CameraStream` | Reads one camera in a background thread, with reconnects |
| `Camera` | Per-camera processing, recording and tile rendering |
| `_make_grid(tiles)` | Arranges the tiles into one window image |
| `FpsMeter` | Measures FPS over a rolling 1-second window |
| `_read_with_retry(cap, url)` | Reads a frame, reconnecting up to 5 times on failure. Returns `(frame, cap)` so the caller always holds the live capture |
| `process_frame(...)` | Resizes the frame and draws the text overlay |
| `_capture_screenshot(frame)` | Saves the current frame as a JPG |
| `_start_recording(frame, fps_est)` / `_stop_recording(writer)` | Opens an MP4 writer at the measured FPS, or closes it |
| `_timestamp()` | Builds the `YYYYMMDD_HHMMSS` string for file names |
| `_close_cap(cap)` / `_cleanup(cap, out)` | Releases resources safely |

## Troubleshooting

- **"Invalid camera URL"**: the address must start with `http://`, `https://` or `rtsp://` and include a host, plus a numeric port if it has one.
- **"camera address not configured"**: `CAMERA_URL` isn't set and no `.env` file was found.
- **`Error number -138` / connection timed out**: the PC can't reach the phone. Check that both are on the same network (not a guest network), the camera app's server is running, the phone's screen is on, and no VPN is active.
- **"Could not connect to the camera"**: check the URL, that the camera app is running, and that both devices are on the same Wi-Fi.
- **"Could not recover the video stream"**: the stream dropped and 5 reconnection attempts failed.
- **"Object detection disabled: ultralytics is not installed"**: run `pip install ultralytics`. The rest of the viewer still works without it.
- **FPS drops a lot with YOLO on**: raise `YOLO_EVERY_N_FRAMES`, lower `YOLO_IMAGE_SIZE`, or press `Y` to switch it off.
- **"Failed to start recording"**: the MP4 writer couldn't open (no write permission, or the codec isn't available). Recording stays off.

## Changelog

- **Automated tests** (`test.py`): TC01-TC10 with a fake camera and evidence images.
- **No overwrites**: `_unique_path()` adds `_2`, `_3`... when a screenshot or recording with the same timestamp already exists.
- **Browser dashboard** (`dashboard.py`, Flask): live MJPEG streams for every camera, per-camera and all-camera controls, an activity log and downloadable saved files.
- **Multiple camera support**: up to 9 streams in a grid, each read in its own thread with its own detectors and recorder. Number keys choose which camera the controls act on. Saved files now include the camera name.
- **Automatic recording on motion** added: starts after sustained motion, includes a 2 s pre-roll and stops after 5 s of quiet, toggled with `A`. Clips are saved as `recordings/motion_*.mp4`.
- **YOLO object detection** added: YOLOv8n draws labelled boxes around detected objects, toggled with `Y`. It's optional, and the app runs without `ultralytics`.
- **Motion detection** added: MOG2 background subtraction highlights moving areas, with a status line and an `M` key to turn it on or off.
- **Screenshots and recordings have their own folders** (`screenshots/`, `recordings/`), created automatically.
- **On-screen feedback**: a 2-second confirmation for screenshots and recording start/stop, and a blinking red REC badge with the elapsed time while recording.
- **URL validation** before connecting catches a missing scheme, a missing host, or a bad port with a clear message.
- **Safe shutdown** reports and saves any recording still running when the program exits on Esc, Ctrl+C or an error.
- **Timeouts now work.** The nonexistent `CAP_PROP_TIMEOUT` was replaced with `CAP_PROP_OPEN_TIMEOUT_MSEC` (5 s) and `CAP_PROP_READ_TIMEOUT_MSEC` (2 s), passed when the capture is created, which is the only way OpenCV applies them.
- **Recordings play at real speed.** The writer uses the measured FPS (limited to 1 to 60, with 15 as the fallback) instead of a fixed 30.
- **A failed recording no longer looks like it started.** `recording` is only set when the writer actually opened.
- **Full config error message.** It now says to set `CAMERA_URL` in the environment or a `.env` file, with an example.
- **FourCC fixed** to lowercase `mp4v`, the standard MP4 code in OpenCV.
- **Rolling FPS.** The overlay shows FPS over the last second instead of the average since startup, so drops show up straight away.
- **Fixed: reconnection never recovered the stream.** `_read_with_retry` opened a new capture on failure but returned only the frame, so `main()` kept reading from the old, released capture and every read after a reconnect failed. It now returns `(frame, cap)`, and `main()` stores the returned capture with `frame, cap = _read_with_retry(cap, ipcam_url)`.
