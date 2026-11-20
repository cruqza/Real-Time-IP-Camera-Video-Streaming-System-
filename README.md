# Phone IP Camera Viewer

A small Python + OpenCV app that streams live video from a phone (or any IP camera) over your local Wi-Fi, shows it in a desktop window with an on-screen info overlay, and lets you take screenshots and record video with single key presses.

## What it does

1. **Finds the camera address** from the `CAMERA_URL` environment variable, or from a `.env` file next to the script. The address is never hardcoded, so credentials stay out of the repo.
2. **Connects to the stream** with OpenCV's FFmpeg backend, using a one-frame buffer to keep latency low.
3. **Shows each frame** in a window called "Phone IP Camera", after:
   - resizing it to 960 px wide (keeping the aspect ratio)
   - drawing a green overlay with the camera name, resolution, current date and time, and an estimated FPS
4. **Reconnects automatically** if the stream drops. It retries up to 5 times, 1 second apart, before giving up. After a successful reconnect the main loop carries on with the new connection.
5. **Handles keyboard controls** for screenshots, recording and exit.
6. **Cleans up** on exit, Ctrl+C or an error: releases the camera, closes any recording file and shuts the window.

## Requirements

- Python 3.7+
- OpenCV with FFmpeg support:

```bash
pip install opencv-python
```

- A phone running an IP camera app (for example "IP Webcam" on Android) or any camera that serves an HTTP/RTSP video stream
- The computer and the camera on the **same Wi-Fi network**

## Setup

Set the camera address in one of two ways.


## Usage

```bash
python camerap.py
```

### Keyboard controls

The video window must be focused.

| Key   | Action                                                              |
|-------|---------------------------------------------------------------------|
| `S`   | Save a screenshot as `screenshot_YYYYMMDD_HHMMSS.jpg`               |
| `R`   | Start or stop recording to `recording_YYYYMMDD_HHMMSS.mp4`          |
| `Esc` | Quit                                                                |

Screenshots and recordings are saved to the folder you run the script from, and they include the overlay.

## How the code is organised

| Function | Purpose |
|----------|---------|
| `main()` | Reads the config, opens the stream, runs the display loop and handles keys |
| `_open_capture(url)` | Opens the stream with a small buffer and a short timeout |
| `_read_with_retry(cap, url)` | Reads a frame, reconnecting up to 5 times on failure. Returns `(frame, cap)` so the caller always holds the live capture |
| `process_frame(...)` | Resizes the frame and draws the text overlay |
| `_capture_screenshot(frame)` | Saves the current frame as a JPG |
| `_start_recording(frame)` / `_stop_recording(writer)` | Opens or closes an MP4 video writer |
| `_timestamp()` | Builds the `YYYYMMDD_HHMMSS` string for file names |
| `_close_cap(cap)` / `_cleanup(cap, out)` | Releases resources safely |

## Troubleshooting

- **"camera address not configured"**: `CAMERA_URL` isn't set and no `.env` file was found.
- **"Could not connect to the camera"**: check the URL, that the camera app is running, and that both devices are on the same Wi-Fi.
- **"Could not recover the video stream"**: the stream dropped and 5 reconnection attempts failed.

