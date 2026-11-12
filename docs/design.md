# Design: Camera Color Picker (DD-CP)

**Version:** 0.2
**Status:** IMPLEMENTED: tested on Linux with the built-in test pattern and in JupyterLab in a headless
browser. **Not yet run on Windows or with a real camera**
**Updated:** 2026-11-11
**Applies to:** `camera_color_picker` v0.2.0, `notebooks/color_picker.ipynb`
**Author:** ticohannan

A Jupyter tool, run from **one notebook cell**, for testing USB cameras and measuring colors. It answers
four questions before any vision code is written: which cameras Windows can see, which resolutions and
frame rates they deliver, what color values a real object gives under real light, and how many pixels
one degree or one millimeter covers. The color answer decides the HSV threshold that will pick out one
block or widget (for example in Robot Demo 4, ECE 6311).

## Changes in 0.2

| Change | Requirement |
| --- | --- |
| Own repository (`ticohannan/camera-color-picker`), split into modules; it has no robot dependency | §11 |
| Video size slider, fit-to-cell option, collapsible control sections, stacked or side-by-side layout, wheel size, so video and controls fit a small laptop screen together | CP-17 |
| Flip and rotate the camera image | CP-18 |
| Show what the driver reports about focus, zoom, exposure and white balance | CP-19 |
| Focal length in pixels and field of view, measured with the crosshair (or from a datasheet field of view) | CP-20 |
| North American spelling | — |

## 1. Environment

| Item | Value |
| --- | --- |
| PC | Windows 10 laptop with a built-in webcam (not used) and up to 3 USB cameras |
| Python | WinPython 3.12, JupyterLab |
| Required | `opencv-python`, `numpy`, `ipywidgets` (8.x), `ipyevents` (mouse events on the video) |
| Optional | `cv2_enumerate_cameras` (camera names and VID:PID), `imageio-ffmpeg` (exact list of camera modes) |

The USB cameras on hand (VID:PID from Windows):

| # | Marking | Build | VID:PID |
| --- | --- | --- | --- |
| 1 | USB2.0 8.0 MEGAPIXEL PC CAMERA | black case, clip | 058F:5608 |
| 2 | hbvcam 3m1917 v22 | bare PCB, black | 058F:3861 |
| 3 | FHD 1080P USBFHD01M | bare PCB, blue | 05A3:9230 |

## 2. Requirements

| ID | Requirement |
| --- | --- |
| CP-1 | List the cameras Windows reports: index, name, VID:PID. Two extra sources: a **test pattern** (no camera needed) and a **still image file** (re-analyze a saved capture). A Refresh button lists them again |
| CP-2 | For the chosen camera, list its modes (format, resolution, frame-rate range). The user chooses resolution and frame rate and applies them. The status line shows what was requested, what the driver reports, and the **measured** capture and display rates |
| CP-3 | Show the live video in the notebook. Display updates are capped (default 15 fps) so the browser never falls behind |
| CP-4 | A crosshair overlay starts at the image center. Pressing the mouse on the video moves it there; dragging moves it continuously |
| CP-5 | The user can type coordinates and press Go, or press Center |
| CP-6 | Origin switch: **corner** (top-left pixel is (0, 0)) or **center** (the pixel at (W/2, H/2) is (0, 0)). In both, x grows to the right and y grows **down** (image convention, as in OpenCV). Typed coordinates use the selected origin; readouts always show both |
| CP-7 | Crosshair color: white, black, red, green, blue, yellow, cyan, magenta, or auto (black or white, whichever contrasts with the image around the crosshair). Line thickness 1–8 px. Optional outline in the opposite shade |
| CP-8 | **Pin-point mode** (default): the crosshair lines leave a gap at the intersection, so the sampled pixel is not covered. One pixel is sampled. Readout: color swatch, both coordinate pairs, RGB, hex, HSV in OpenCV units (H 0–179, S and V 0–255), HSV in degrees and percent, HSV in MATLAB units (0–1) |
| CP-9 | **Cluster mode**: a circle of user-set diameter, sampled at N points. Pattern: random ("shotgun", fixed seed, Re-scatter button for a new draw) or even (sunflower spiral). Readout gives the coordinate range covered |
| CP-10 | Cluster statistics, per channel H, S, V, R, G, B: mean, standard deviation, min, max. Hue uses circular statistics (§5), so a red that straddles 0/180 averages to red, not cyan. Swatch of the mean color. A copy-ready OpenCV `inRange` lower/upper pair from the min/max; two pairs when the hue range wraps through 0 |
| CP-11 | Optional exclusion of points that differ from the mean by more than k standard deviations in any of H, S, V (k slider, default 2.0) |
| CP-12 | A color wheel (hue = angle, saturation = radius) with every sample drawn in its own color: kept points ringed in black, excluded ones ringed in red, the mean marked, the kept range outlined; a brightness (V) bar beside it |
| CP-13 | Capture button: saves the frame as shown (after CP-18) without overlay as PNG, optionally also the annotated view, plus a JSON file with the camera, mode, orientation, crosshair, settings, camera properties, focal estimate and the current readout |
| CP-14 | Freeze: stop taking new frames. Hold readout: stop refreshing the text so it can be selected and copied |
| CP-15 | It all runs from one cell. Running the cell again closes the previous picker and releases its camera. Stop releases the camera. Errors appear in the status line and never kill the notebook |
| CP-16 | Camera settings button: opens the Windows (DirectShow) property dialog of the open camera, to lock exposure and white balance |
| CP-17 | **Size and layout.** Video width slider (160–1920 px, default 800). "Fit to cell" scales the video to the notebook's width instead. Layout: **Stacked** (controls under the video) or **Side by side** (video on the left; readout and controls on the right in a column that scrolls on its own). Control groups (Source and mode, Sampling, Crosshair and orientation, Lens and focal length, Capture) are collapsible. Color wheel size slider |
| CP-18 | **Orientation.** Rotate 0°, 90°, 180°, 270° clockwise; flip horizontal (mirror left-right); flip vertical. Applied to every frame as it arrives (flips first, then the rotation), so the video, crosshair, sampling, measurements and captures all use the turned image. A 90° or 270° turn swaps width and height and re-centers the crosshair. The JSON records the orientation and the crosshair in the camera's own (unturned) pixel coordinates too |
| CP-19 | **Camera properties.** After Start, and on a Read button: the backend name and what the driver reports for focus, autofocus, zoom, exposure, auto-exposure, gain, brightness, contrast, saturation, white balance, auto white balance, sharpness, backlight compensation. Values the driver does not support show as "not reported" |
| CP-20 | **Focal length and field of view.** UVC webcams do not report their lens focal length (§9), so it is measured: put the crosshair on two points of an object of known length L, set them as A and B, enter L and the distance Z from the camera to the object. Or enter a datasheet diagonal field of view. Shows f in pixels at the current resolution, horizontal / vertical / diagonal field of view, pixels per degree at the center, and f in millimeters when a pixel pitch is entered |

## 3. Layout

Stacked (default):

```
Camera color picker v0.2   layout (Stacked | Side by side)  video width [—o—]  [x] fit to cell
status: requested 1280×720 MJPG @30 | driver ... | measured capture 29.8 fps, display 15.0 fps
┌──────────────── video (crosshair overlay, drag with the mouse) ────────────────┐
└────────────────────────────────────────────────────────────────────────────────┘
┌ swatch ┐ ┌ copyable values ┐ ┌ color wheel + V bar ┐     [Freeze] [Hold readout]
cluster table
▸ Source and mode          (camera, Refresh, Start, Stop, Camera settings, file, mode, fps, Apply, display fps)
▸ Sampling                 (Pin-point | Cluster, gap, diameter, points, pattern, Re-scatter, outliers, k)
▸ Crosshair and orientation (color, thickness, outline, info text, origin, x, y, Go, Center, rotate, flips)
▸ Lens and focal length    (camera properties, A/B points, L, Z, datasheet FOV, pixel pitch, results)
▸ Capture                  (folder, also annotated, Capture still)
```

Side by side: the video on the left; on the right, in a column of 85 % of the window height that
scrolls on its own: the readout, then the same collapsible groups. "Source and mode" starts open, the
others closed.

## 4. Cameras and modes

**Listing (CP-1).** On Windows the camera is opened through DirectShow (`cv2.CAP_DSHOW`), so the list must
be in DirectShow order. In order of preference:

1. `cv2_enumerate_cameras.enumerate_cameras(cv2.CAP_DSHOW)`: index, name, device path, VID:PID.
2. `pygrabber` (names only), if installed.
3. Probe indices 0–5 by opening each (slow, names unknown: "Camera n").

**Modes (CP-2).** OpenCV cannot ask a camera what it supports. In order of preference:

1. FFmpeg's DirectShow listing, which reads the camera's own capability table:
   `ffmpeg -hide_banner -f dshow -list_options true -video_device_number n -i video=<name>`
   (`n` separates cameras with the same name). The FFmpeg program comes with `imageio-ffmpeg`. Lines such as
   `vcodec=mjpeg  min s=1280x720 fps=5 max s=1280x720 fps=30` become modes
   (format MJPG, 1280×720, 5–30 fps); `pixel_format=yuyv422` becomes YUY2.
2. Probing: set each common resolution (160×120 … 3264×2448) and keep what the driver reports back, once
   with the default format and once with MJPG.

Frame-rate choices are the standard rates (5, 7.5, 10, 15, 20, 24, 25, 30, 50, 60, 90, 120) inside the
mode's range, plus its maximum. **Apply** reopens the camera and sets, in this order, FOURCC, width,
height, FPS; if the format read back differs, it tries width and height first. A USB 2.0 camera usually
reaches full rate at high resolution only in MJPG.

## 5. Sampling and statistics

Coordinates are held in full-resolution pixels of the turned image (CP-18); the display is a scaled copy.
A mouse position on the display maps to `x = round(x_display / scale)`, clamped to the frame.

Corner ↔ center: `xc = x − W//2`, `yc = y − H//2` (y down in both).

**Cluster points.** For diameter D (radius R = D/2) and N points:
random: r = R·√u, φ = 2πw with u, w uniform (√u makes the density uniform over the disc);
even: the k-th point at r = R·√((k + 0.5)/N), φ = k·137.508°. Offsets are rounded to pixels, duplicates
removed, and points outside the frame dropped.

**Hue is an angle.** OpenCV hue h (0–179) is the angle 2h degrees. With a_i = 2h_i in radians:

```
C = mean(cos a_i),  S = mean(sin a_i)
mean angle = Atan2(x = C, y = S)        Python: math.atan2(S, C)  (y first)
R = √(C² + S²)                           (1 = all hues equal, 0 = spread around the circle)
circular std = √(−2 ln R)  radians      reported in OpenCV hue units (× 90/π)
hue difference d(h, m) = ((h − m + 90) mod 180) − 90       in −90 … 89
```

**Hue range.** The smallest arc that contains every kept hue: sort the distinct hues around the circle,
find the largest gap between neighbors (including the gap from the last back to the first, + 180);
the range runs from the hue after that gap to the hue before it. If it wraps (low > high), the OpenCV
threshold needs two `inRange` calls OR-ed: `[lo, 179]` and `[0, hi]`.

**Outliers (CP-11).** From all samples: mean and std of H (circular), S, V. A point is excluded if
|d(h, mean)|/σ_H > k or |s − mean|/σ_S > k or |v − mean|/σ_V > k, with each σ floored at 1 unit. One
pass. If that would exclude everything, nothing is excluded. Statistics are then recomputed on the kept
points.

**Low saturation.** When the mean S of the kept points is below 40, hue is unreliable (grays, white,
black) and the readout says so.

## 6. Color wheel (CP-12)

A disc of radius R (wheel size slider, CP-17; default 130 px): the pixel at angle φ (counterclockwise from
the right, image y up) and radius r is HSV(φ/2, 255·r/R, 255). A sample with (h, s) sits at φ = 2h,
r = R·s/255. The kept range is outlined as the sector from h_lo to h_hi between s_min and s_max. The V
bar (0 at the bottom, 255 at the top) has a tick per sample and a bracket for v_min–v_max.

## 7. Orientation (CP-18)

With the camera frame of width w and height h, the flips are applied first, then the clockwise rotation:

| Step | Pixel (x, y) moves to | New size |
| --- | --- | --- |
| flip horizontal | (w − 1 − x, y) | w × h |
| flip vertical | (x, h − 1 − y) | w × h |
| rotate 90° | (h − 1 − y, x) | h × w |
| rotate 180° | (w − 1 − x, h − 1 − y) | w × h |
| rotate 270° | (y, w − 1 − x) | h × w |

(OpenCV: `cv2.flip(img, 1)`, `cv2.flip(img, 0)`, `cv2.rotate` with `ROTATE_90_CLOCKWISE`, `ROTATE_180`,
`ROTATE_90_COUNTERCLOCKWISE`.) The inverse map gives the crosshair in camera pixels for the JSON.
A mirror (a camera looking through a mirror, or a front camera) is one flip; a camera mounted upside down
is a 180° rotation.

## 8. Camera properties (CP-19)

Read with `cap.get(...)` for `CAP_PROP_FOCUS`, `AUTOFOCUS`, `ZOOM`, `EXPOSURE`, `AUTO_EXPOSURE`, `GAIN`,
`BRIGHTNESS`, `CONTRAST`, `SATURATION`, `WB_TEMPERATURE`, `AUTO_WB`, `SHARPNESS`, `BACKLIGHT`, plus
`cap.getBackendName()`. DirectShow returns the driver's raw units (exposure is usually log2 seconds,
e.g. −6 = 1/64 s; focus is a motor position, not a distance). A value of −1, or a failed call, is shown
as "not reported". A fixed-focus lens reports nothing for focus.

## 9. Focal length and field of view (CP-20)

A UVC webcam's lens focal length is not available through DirectShow or OpenCV: the UVC camera terminal
can describe it, but drivers rarely fill it in and Windows does not pass it on. What matters for vision
is the focal length **in pixels**, f, at the resolution in use (pinhole model). Two ways to get it:

**Measured.** An object of known length L (mm) faces the camera at distance Z (mm) from the lens, near
the image center. Crosshair on one end → **Set A**; on the other end → **Set B**. With p the pixel
distance from A to B:

```
f = p · Z / L                                  (similar triangles: p / f = L / Z)
FOV_h = 2 · Atan2(x = f, y = W/2)              Python: 2 * math.atan2(W / 2, f)
FOV_v = 2 · Atan2(x = f, y = H/2),  FOV_d = 2 · Atan2(x = f, y = √(W² + H²)/2)
pixels per degree at the center = f · π/180
```

**From a datasheet.** A diagonal field of view D (deg) gives f = (√(W² + H²)/2) / tan(D/2).

With a pixel pitch P (µm) at this resolution: f_mm = f · P / 1000. f scales with the image width only
when a mode uses the full sensor; modes that crop (common at 1280×720 on 4:3 sensors) need their own
measurement, so f is always reported with its resolution. The result is approximate: lens distortion
and an object off the center change it by a few percent; a checkerboard calibration
(`cv2.calibrateCamera`) is the accurate method and is out of scope.

## 10. Captures (CP-13)

| File | Content |
| --- | --- |
| `<stem>.png` | the frame as shown (after CP-18), no overlay |
| `<stem>_annotated.png` | the same frame with the overlay (optional) |
| `<stem>.json` | time, source (name, VID:PID, index), mode requested and reported, orientation, crosshair (corner, center, camera pixels), sampling settings, camera properties, focal estimate, the readout text and the cluster statistics |

`<stem>` = `YYYYMMDD_HHMMSS_mmm_<source label, letters and digits only>`.

## 11. Code layout and threads

| Module | Contents |
| --- | --- |
| `camera.py` | sources (DirectShow cameras, test pattern, image file), camera listing, modes, properties |
| `analysis.py` | coordinates, orientation, color statistics, sampling, focal length |
| `drawing.py` | crosshair, measuring line, color wheel, image encoding |
| `readout.py` | the text and table shown under the video |
| `widget.py` | `ColorPicker`, the notebook interface |

One background thread per picker: read a frame (blocks at the camera's rate), turn it (CP-18), and if the
display interval has passed, draw the overlay on a scaled copy, encode it as JPEG and assign it to the
`Image` widget. Statistics and the color wheel are recomputed at most 4 times a second. Mouse and button
callbacks run in the kernel's main thread and only change settings. Changing the source or mode stops the
thread, reopens the camera and starts it again. Without `ipyevents` the picker still runs; the crosshair
is then moved only with typed coordinates.

## 12. Out of scope

Threshold mask preview and blob detection (they belong to the tracking code that reads the JSON);
lens distortion calibration.

## 13. Tests (`python -m pytest -q`)

| Requirement | Test (`tests/test_color_picker.py`) |
| --- | --- |
| CP-1 | `test_sources_listed` |
| CP-2 | `test_parse_dshow_options`, `test_fps_choices` |
| CP-4, CP-5, CP-6 | `test_coords_roundtrip`, `test_ui_mouse_and_go` |
| CP-7 | `test_overlay_gap_and_circle` |
| CP-8 | `test_pin_readout_known_pixel` |
| CP-9 | `test_disk_offsets` |
| CP-10 | `test_circular_hue_stats_wrap`, `test_hue_arc_and_inrange_split`, `test_cluster_on_two_colors` |
| CP-11 | `test_outlier_exclusion` |
| CP-12 | `test_wheel_point_positions`, `test_wheel_sizes` |
| CP-13 | `test_capture_files` |
| CP-14, CP-15 | `test_ui_freeze_hold_and_rerun` |
| CP-17 | `test_ui_size_and_layout` |
| CP-18 | `test_orient_image_and_points`, `test_ui_orientation` |
| CP-19 | `test_camera_properties` |
| CP-20 | `test_focal_from_measurement`, `test_focal_from_fov`, `test_ui_measure_focal` |
| One cell runs | `test_notebook_executes` |
