# camera-color-picker

A Jupyter tool, run from one notebook cell, for testing USB cameras and measuring colors on Windows.

- Lists the cameras Windows sees (name, VID:PID), their modes (format, resolution, frame-rate range), and
  shows the requested, reported and **measured** frame rates.
- Live video with a draggable crosshair: **pin-point** (one pixel) or **cluster** (a circle of N points).
- Readouts in RGB, hex and HSV (OpenCV, degrees/percent, MATLAB 0–1); cluster statistics with circular
  hue, outlier exclusion, a color wheel and copy-ready OpenCV `inRange` ranges (split when red wraps
  through hue 0).
- Flip and rotate the image; stacked or side-by-side layout and a video size slider for small screens.
- Camera properties the driver reports, and the **focal length in pixels and field of view**, measured with
  two crosshair points on an object of known size.
- Still capture: PNG (plain and annotated) plus a JSON file with every setting and the readout.

Design document: [`docs/design.md`](docs/design.md). Status: tested on Linux with the built-in test
pattern; not yet run on Windows with a real camera.

## Install (Windows, WinPython)

1. Open the **WinPython Command Prompt**.
2. `pip install opencv-python numpy ipywidgets ipyevents cv2_enumerate_cameras imageio-ffmpeg`
3. Close JupyterLab completely and start it again (ipyevents adds a browser extension; without it the
   crosshair cannot be dragged, only typed).
4. Get the code: `git clone https://github.com/ticohannan/camera-color-picker.git`
   (or unzip the release into a folder in your Jupyter tree).

`cv2_enumerate_cameras` gives camera names and VID:PID; `imageio-ffmpeg` gives the exact list of camera
modes. Both are optional: without them the cameras are numbered and the modes are probed.

## Run

Open `notebooks/color_picker.ipynb` and run its one cell:

```python
import sys, pathlib
REPO = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(REPO))
from camera_color_picker import ColorPicker

picker = ColorPicker(save_dir=REPO / "captures", display_width=800, layout="stacked")   # or layout="side"
picker.show();
```

Close other programs that use the camera first (Camera app, Teams, Zoom). Run the cell again to restart;
the previous camera is released. Pick the **Test pattern** source to try everything without a camera.

## Tests

```
pip install pytest nbclient nbformat ipykernel
python -m pytest -q
```

## License

BSD 3-Clause. Copyright (c) 2026, ticohannan.
