"""The notebook interface (CP-1 ... CP-20). One `ColorPicker` per cell; running the cell again replaces it."""
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .analysis import (ROTATIONS, center_to_corner, clamp_xy, cluster_stats, corner_to_center, describe_pixel,
                       disk_offsets, focal_from_dfov, focal_from_measurement, orient, oriented_size,
                       sample_points, unorient_point)
from .camera import CameraSource, FileSource, Mode, SourceEntry, TestPatternSource, fps_choices, list_sources
from .drawing import COLORS, auto_color, draw_crosshair, draw_measure, encode, render_wheel
from .readout import cluster_text, focal_text, pin_text, props_html, stats_html

__all__ = ["ColorPicker"]
_ACTIVE = []


def _version():
    from . import __version__
    return __version__


class ColorPicker:
    """Camera color picker for Jupyter.

    ColorPicker(save_dir="captures", display_width=800, layout="stacked").show()
    layout: "stacked" (controls under the video) or "side" (controls beside it, for short screens).
    """

    def __init__(self, save_dir="captures", display_width=800, display_fps=15, layout="stacked",
                 list_cams=True, wheel_size=260):
        import ipywidgets as W
        for old in list(_ACTIVE):                       # CP-15: running the cell again releases the camera
            try:
                old.close()
            except Exception:
                pass
        _ACTIVE.clear()
        _ACTIVE.append(self)
        self.W = W
        self.save_dir = Path(save_dir)
        self.src = None
        self.entry: Optional[SourceEntry] = None
        self.modes: List[Mode] = []
        self.modes_from = ""
        self.info = {}
        self.props = {}
        self.raw = None                       # latest frame as the camera delivered it
        self.frame = None                     # the same frame turned (CP-18); everything works on this
        self._frame_key = None
        self.fw, self.fh = 640, 480
        self.x, self.y = 320, 240
        self.A = self.B = None                # focal-length measurement points (CP-20)
        self.requested = ""
        self.cap_fps = 0.0
        self.disp_fps = 0.0
        self._offsets = disk_offsets(300, 41)
        self._seed = 0
        self._last_stats = None
        self._last_pts = None
        self._last_keep = None
        self._last_text = ""
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._fails = 0
        self.events = None
        self._build(display_width, display_fps, list_cams, layout, wheel_size)

    # ------------------------------------------------------------- layout (CP-17)
    def _build(self, display_width, display_fps, list_cams, layout, wheel_size):
        W = self.W
        L = W.Layout
        self.sources = list_sources(list_cams)

        # top bar
        self.w_layout = W.ToggleButtons(options=[("Stacked", "stacked"), ("Side by side", "side")],
                                        value="side" if layout == "side" else "stacked",
                                        style={"button_width": "100px"}, layout=L(width="auto"))
        self.w_vwidth = W.IntSlider(value=int(display_width), min=160, max=1920, step=16, description="video width",
                                    continuous_update=False, layout=L(width="340px"),
                                    style={"description_width": "80px"})
        self.w_fit = W.Checkbox(value=False, description="fit to cell", indent=False, layout=L(width="100px"))
        self.w_wsize = W.IntSlider(value=int(wheel_size), min=120, max=320, step=20, description="wheel",
                                   continuous_update=False, layout=L(width="260px"),
                                   style={"description_width": "50px"})
        self.w_status = W.HTML("<i>choose a source and press Start</i>")
        self.w_video = W.Image(format="jpeg", layout=L(max_width="100%", height="auto"))

        # source and mode
        self.w_src = W.Dropdown(options=[(s.label, i) for i, s in enumerate(self.sources)], description="source",
                                layout=L(width="420px", max_width="100%"))
        self.w_refresh = W.Button(description="Refresh", layout=L(width="80px"))
        self.w_start = W.Button(description="Start", button_style="success", layout=L(width="80px"))
        self.w_stop = W.Button(description="Stop", button_style="danger", layout=L(width="80px"))
        self.w_settings = W.Button(description="Camera settings", layout=L(width="140px"),
                                   tooltip="Windows camera property dialog (exposure, white balance)")
        self.w_path = W.Text(value="", placeholder="path of a .png/.jpg (for 'Still image file')",
                             description="file", layout=L(width="520px", max_width="100%"))
        self.w_mode = W.Dropdown(options=[], description="mode", layout=L(width="330px", max_width="100%"))
        self.w_fps = W.Dropdown(options=[], description="fps", layout=L(width="150px"))
        self.w_apply = W.Button(description="Apply", layout=L(width="80px"))
        self.w_dfps = W.IntSlider(value=display_fps, min=2, max=30, description="display fps",
                                  layout=L(width="300px"))

        # sampling
        self.w_samp = W.ToggleButtons(options=["Pin-point", "Cluster"], value="Pin-point", description="sample",
                                      style={"button_width": "90px"})
        self.w_gap = W.IntSlider(value=6, min=1, max=40, description="pin gap px", layout=L(width="300px"))
        self.w_diam = W.IntSlider(value=41, min=3, max=401, step=2, description="diameter px",
                                  layout=L(width="320px"))
        self.w_npts = W.IntSlider(value=300, min=10, max=3000, step=10, description="points",
                                  layout=L(width="320px"))
        self.w_pattern = W.Dropdown(options=[("random (shotgun)", "random"), ("even (spiral)", "even")],
                                    value="random", description="pattern", layout=L(width="240px"))
        self.w_rescatter = W.Button(description="Re-scatter", layout=L(width="100px"))
        self.w_showpts = W.Checkbox(value=True, description="show points", indent=False, layout=L(width="110px"))
        self.w_exclude = W.Checkbox(value=False, description="exclude outliers", indent=False,
                                    layout=L(width="140px"))
        self.w_k = W.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.1, description="k (sd)",
                                 layout=L(width="300px"))

        # crosshair and orientation
        self.w_color = W.Dropdown(options=["auto"] + list(COLORS), value="auto", description="line color",
                                  layout=L(width="200px"))
        self.w_thick = W.IntSlider(value=1, min=1, max=8, description="thickness", layout=L(width="260px"))
        self.w_outline = W.Checkbox(value=True, description="outline", indent=False, layout=L(width="80px"))
        self.w_hud = W.Checkbox(value=True, description="info text in video", indent=False, layout=L(width="140px"))
        self.w_origin = W.ToggleButtons(options=["Corner", "Center"], value="Corner", description="origin",
                                        style={"button_width": "70px"})
        self.w_x = W.IntText(value=0, description="x", layout=L(width="130px"))
        self.w_y = W.IntText(value=0, description="y", layout=L(width="130px"))
        self.w_go = W.Button(description="Go", layout=L(width="60px"))
        self.w_center = W.Button(description="Center", layout=L(width="80px"))
        self.w_rot = W.Dropdown(options=[(f"{r}° cw", r) for r in ROTATIONS], value=0, description="rotate",
                                layout=L(width="170px"))
        self.w_fliph = W.Checkbox(value=False, description="flip horizontal (mirror)", indent=False,
                                  layout=L(width="180px"))
        self.w_flipv = W.Checkbox(value=False, description="flip vertical", indent=False, layout=L(width="110px"))

        # lens and focal length
        self.w_readprops = W.Button(description="Read camera properties", layout=L(width="190px"))
        self.w_props = W.HTML(props_html({}))
        self.w_setA = W.Button(description="Set A", layout=L(width="70px"), tooltip="A = crosshair position")
        self.w_setB = W.Button(description="Set B", layout=L(width="70px"), tooltip="B = crosshair position")
        self.w_clearAB = W.Button(description="Clear", layout=L(width="70px"))
        st = {"description_width": "150px"}
        self.w_len = W.FloatText(value=0.0, description="L, A to B (mm)", style=st, layout=L(width="250px"))
        self.w_dist = W.FloatText(value=0.0, description="Z, lens to object (mm)", style=st, layout=L(width="250px"))
        self.w_dfov = W.FloatText(value=0.0, description="diagonal FOV (deg)", style=st,
                                  layout=L(width="250px"))
        self.w_pitch = W.FloatText(value=0.0, description="pixel pitch (µm)", style=st,
                                   layout=L(width="250px"))
        self.w_focal = W.Textarea(value="", layout=L(width="100%", height="150px"))

        # capture
        self.w_dir = W.Text(value=str(self.save_dir), description="save to", layout=L(width="420px", max_width="100%"))
        self.w_annot = W.Checkbox(value=True, description="also annotated", indent=False, layout=L(width="130px"))
        self.w_capture = W.Button(description="Capture still", button_style="info", icon="camera",
                                  layout=L(width="140px"))
        self.w_saved = W.HTML("")

        # readout
        self.w_freeze = W.ToggleButton(value=False, description="Freeze frame", layout=L(width="120px"))
        self.w_hold = W.ToggleButton(value=False, description="Hold readout", layout=L(width="120px"),
                                     tooltip="stop refreshing the text so it can be selected and copied")
        self.w_swatch = W.HTML(self._swatch_html((128, 128, 128), "—"))
        self.w_text = W.Textarea(value="", layout=L(width="100%", height="170px"))
        self.w_wheel = W.Image(format="png", layout=W.Layout(flex="0 0 auto", width="auto", height="auto", object_fit="contain"))
        self.w_table = W.HTML("")

        HB = lambda ch: W.HBox(ch, layout=L(flex_flow="row wrap", align_items="center"))
        self.cluster_box = W.VBox([HB([self.w_diam, self.w_npts]),
                                   HB([self.w_pattern, self.w_rescatter, self.w_showpts]),
                                   HB([self.w_exclude, self.w_k])])
        self.pin_box = HB([self.w_gap])

        def section(title, children, open_=False):
            a = W.Accordion(children=[W.VBox(children)], titles=(title,))
            a.selected_index = 0 if open_ else None
            return a

        self.sections = [
            section("Source and mode", [
                HB([self.w_src, self.w_refresh, self.w_start, self.w_stop, self.w_settings]), self.w_path,
                HB([self.w_mode, self.w_fps, self.w_apply, self.w_dfps])], open_=True),
            section("Sampling", [self.w_samp, self.pin_box, self.cluster_box]),
            section("Crosshair and orientation", [
                HB([self.w_color, self.w_thick, self.w_outline, self.w_hud]),
                HB([self.w_origin, self.w_x, self.w_y, self.w_go, self.w_center]),
                HB([self.w_rot, self.w_fliph, self.w_flipv])]),
            section("Lens and focal length", [
                self.w_readprops, self.w_props,
                HB([self.w_setA, self.w_setB, self.w_clearAB]),
                HB([self.w_len, self.w_dist]), HB([self.w_dfov, self.w_pitch]), self.w_focal]),
            section("Capture", [HB([self.w_dir, self.w_annot, self.w_capture]), self.w_saved]),
        ]
        self.topbar = HB([W.HTML(f"<b>Camera color picker</b> v{_version()}&nbsp;&nbsp;"), self.w_layout,
                          self.w_vwidth, self.w_fit, self.w_wsize])
        self.body = W.Box()
        self.widget = W.VBox([self.topbar, self.w_status, self.body])
        self._apply_layout()
        self._size_video()
        self._show_mode_boxes()
        self._update_focal()
        self.w_wheel.value = encode(render_wheel(R=self._wheel_r()), ".png")

        self.w_layout.observe(lambda c: self._apply_layout(), "value")
        self.w_vwidth.observe(lambda c: self._size_video(), "value")
        self.w_fit.observe(lambda c: self._size_video(), "value")
        self.w_wsize.observe(lambda c: self._redraw_now(), "value")
        self.w_refresh.on_click(lambda b: self._refresh())
        self.w_start.on_click(lambda b: self._safe(self.start))
        self.w_stop.on_click(lambda b: self._safe(self.stop))
        self.w_apply.on_click(lambda b: self._safe(self.apply_mode))
        self.w_settings.on_click(lambda b: self._safe(self._settings))
        self.w_mode.observe(lambda c: self._fill_fps(), "value")
        self.w_samp.observe(lambda c: self._show_mode_boxes(), "value")
        for wdg in (self.w_diam, self.w_npts, self.w_pattern):
            wdg.observe(lambda c: self._new_offsets(), "value")
        self.w_rescatter.on_click(lambda b: self._new_offsets(reseed=True))
        self.w_go.on_click(lambda b: self._go())
        self.w_center.on_click(lambda b: self.set_xy(self.fw // 2, self.fh // 2))
        self.w_origin.observe(lambda c: self._sync_xy_boxes(), "value")
        for wdg in (self.w_rot, self.w_fliph, self.w_flipv):
            wdg.observe(lambda c: self._orientation_changed(), "value")
        self.w_readprops.on_click(lambda b: self._safe(self.read_properties))
        self.w_setA.on_click(lambda b: self._set_point("A"))
        self.w_setB.on_click(lambda b: self._set_point("B"))
        self.w_clearAB.on_click(lambda b: self._set_point(None))
        for wdg in (self.w_len, self.w_dist, self.w_dfov, self.w_pitch):
            wdg.observe(lambda c: self._update_focal(), "value")
        self.w_capture.on_click(lambda b: self._safe(self.capture))
        try:
            from ipyevents import Event
            self.events = Event(source=self.w_video, watched_events=["mousedown", "mousemove"],
                                prevent_default_action=True, wait=25, throttle_or_debounce="throttle")
            self.events.on_dom_event(self._on_mouse)
        except Exception as e:
            self.events = None
            self._status(f"ipyevents not available ({e.__class__.__name__}): drag disabled, use x/y and Go")

    def _apply_layout(self):
        W, L = self.W, self.W.Layout
        tools = W.HBox([self.w_freeze, self.w_hold])
        if self.w_layout.value == "side":
            # inner column keeps its natural height (no flex shrink); the outer box scrolls
            inner = W.VBox([tools, self.w_swatch, self.w_text, self.w_wheel, self.w_table] + self.sections,
                           layout=L(flex="0 0 auto", width="100%"))
            right = W.VBox([inner], layout=L(flex="1 1 360px", min_width="320px", max_height="85vh",
                                             overflow="auto", padding="0 0 0 8px"))
            left = W.VBox([self.w_video], layout=L(flex="0 1 auto", max_width="70%"))
            self.body.children = [left, right]
            self.body.layout = L(display="flex", flex_flow="row", align_items="flex-start", width="100%")
        else:
            readout = W.HBox([W.VBox([tools, self.w_swatch, self.w_text], layout=L(flex="1 1 360px")),
                              self.w_wheel], layout=L(flex_flow="row wrap", align_items="flex-start"))
            self.body.children = [self.w_video, readout, self.w_table] + self.sections
            self.body.layout = L(display="flex", flex_flow="column", width="100%")

    def _size_video(self):
        self.w_video.layout.width = "100%" if self.w_fit.value else f"{self.w_vwidth.value}px"

    def _wheel_r(self):
        return max(int(self.w_wsize.value) // 2, 40)

    def show(self):
        from IPython.display import display
        display(self.widget)
        return self

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _swatch_html(bgr, caption):
        b, g, r = bgr
        return (f"<div style='display:flex;align-items:center;gap:10px'>"
                f"<div style='width:80px;height:40px;border:1px solid #444;background:rgb({r},{g},{b})'></div>"
                f"<span style='font-family:monospace'>{caption}</span></div>")

    def _status(self, msg, err=False):
        col = "#c00" if err else "inherit"
        self.w_status.value = f"<span style='color:{col}'>{msg}</span>"

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            self._status(f"{e.__class__.__name__}: {e}", err=True)

    def _show_mode_boxes(self):
        cl = self.w_samp.value == "Cluster"
        self.cluster_box.layout.display = "" if cl else "none"
        self.pin_box.layout.display = "none" if cl else ""

    def _refresh(self):
        self._status("listing cameras ...")
        self.sources = list_sources(True)
        self.w_src.options = [(s.label, i) for i, s in enumerate(self.sources)]
        self._status(f"{len(self.sources) - 2} camera(s) found")

    def _new_offsets(self, reseed=False):
        if reseed:
            self._seed += 1
        self._offsets = disk_offsets(self.w_npts.value, self.w_diam.value, self.w_pattern.value, self._seed)

    def _fill_fps(self):
        i = self.w_mode.value
        if i is None or i >= len(self.modes):
            self.w_fps.options = []
            return
        ch = fps_choices(self.modes[i])
        self.w_fps.options = [(f"{f:g}", f) for f in ch]
        self.w_fps.value = 30 if 30 in ch else ch[-1]

    def _redraw_now(self):
        if self.frame is not None:
            self.step_stats(force=True)
        else:
            self.w_wheel.value = encode(render_wheel(R=self._wheel_r()), ".png")

    # ------------------------------------------------------------- crosshair (CP-4 ... CP-6)
    def set_xy(self, x, y):
        self.x, self.y = clamp_xy(x, y, self.fw, self.fh)
        self._sync_xy_boxes()

    def _sync_xy_boxes(self):
        if self.w_origin.value == "Center":
            xs, ys = corner_to_center(self.x, self.y, self.fw, self.fh)
        else:
            xs, ys = self.x, self.y
        self.w_x.value, self.w_y.value = int(xs), int(ys)

    def _go(self):
        x, y = self.w_x.value, self.w_y.value
        if self.w_origin.value == "Center":
            x, y = center_to_corner(x, y, self.fw, self.fh)
        self.set_xy(x, y)

    def _scale(self):
        return self.w_vwidth.value / float(self.fw)

    def _on_mouse(self, ev):
        self.last_mouse = ev                   # kept for troubleshooting
        if ev.get("type") == "mousemove" and not (ev.get("buttons", 0) & 1):
            return
        if "dataX" in ev:                    # pixel of the displayed (scaled) image
            dx, dy = ev["dataX"], ev["dataY"]
        else:
            k = self.w_vwidth.value / max(ev.get("boundingRectWidth", 1), 1)
            dx, dy = ev["relativeX"] * k, ev["relativeY"] * k
        s = self._scale()
        self.set_xy(dx / s, dy / s)

    # ------------------------------------------------------------- orientation (CP-18)
    def _orient_key(self):
        return (self.w_rot.value, self.w_fliph.value, self.w_flipv.value)

    def _set_frame(self, raw):
        """Turn a raw frame and make it current; re-center the crosshair if the size changed."""
        f = orient(raw, *self._orient_key())
        with self._lock:
            self.raw = raw
            self.frame = f
            self._frame_key = self._orient_key()
        if (f.shape[1], f.shape[0]) != (self.fw, self.fh):
            self.fh, self.fw = f.shape[:2]
            self.A = self.B = None
            self.set_xy(self.fw // 2, self.fh // 2)
            self._update_focal()

    def _orientation_changed(self):
        if self.raw is not None:
            self._set_frame(self.raw)          # also turns a frozen frame
        else:
            w, h = oriented_size(*self._camera_size(), self.w_rot.value)
            self.fw, self.fh = w, h
            self.set_xy(w // 2, h // 2)
        self.A = self.B = None
        self._update_focal()

    def _camera_size(self):
        if self.raw is not None:
            return self.raw.shape[1], self.raw.shape[0]
        return self.info.get("w") or 640, self.info.get("h") or 480

    # ------------------------------------------------------------- lens (CP-19, CP-20)
    def read_properties(self):
        self.props = self.src.properties() if self.src is not None else {}
        self.w_props.value = props_html(self.props)

    def _set_point(self, which):
        if which == "A":
            self.A = (self.x, self.y)
        elif which == "B":
            self.B = (self.x, self.y)
        else:
            self.A = self.B = None
        self._update_focal()

    def focal(self) -> dict:
        p = math.hypot(self.B[0] - self.A[0], self.B[1] - self.A[1]) if self.A and self.B else 0.0
        f_meas = focal_from_measurement(p, self.w_len.value, self.w_dist.value) if p else None
        f_sheet = focal_from_dfov(self.w_dfov.value, self.fw, self.fh) if self.w_dfov.value else None
        return {"frame": [self.fw, self.fh], "A": self.A, "B": self.B, "ab_px": round(p, 2),
                "L_mm": self.w_len.value, "Z_mm": self.w_dist.value, "f_px_measured": f_meas,
                "datasheet_dfov_deg": self.w_dfov.value or None, "f_px_datasheet": f_sheet,
                "pixel_pitch_um": self.w_pitch.value or None}

    def _update_focal(self):
        f = self.focal()
        self.w_focal.value = focal_text(self.fw, self.fh, self.A, self.B, self.w_len.value, self.w_dist.value,
                                        f["f_px_measured"], self.w_dfov.value, f["f_px_datasheet"],
                                        self.w_pitch.value)

    # ------------------------------------------------------------- source control (CP-1, CP-2)
    def start(self):
        self.stop()
        entry = self.sources[self.w_src.value]
        if entry.kind == "camera":
            src = CameraSource(entry)
        elif entry.kind == "test":
            src = TestPatternSource(entry)
        else:
            src = FileSource(entry, self.w_path.value.strip().strip('"'))
        self._status(f"reading modes of {entry.label} ...")
        modes, how = src.modes()                   # before opening: FFmpeg lists modes of a free camera
        self._status(f"opening {entry.label} ...")
        info = src.open()
        self.src, self.entry = src, entry
        self.modes, self.modes_from = modes, how
        self.w_mode.options = [(m.label, i) for i, m in enumerate(modes)]
        cur = next((i for i, m in enumerate(modes) if (m.w, m.h) == (info["w"], info["h"])), None)
        if cur is not None:
            self.w_mode.value = cur
        self._fill_fps()
        self.requested = "driver default"
        self._begin(info)

    def apply_mode(self):
        if self.src is None:
            return self.start()
        i = self.w_mode.value
        if i is None:
            return
        m, f = self.modes[i], self.w_fps.value
        self._halt()
        info = self.src.open(m, f)
        self.requested = f"{m.fmt or 'default'} {m.w}x{m.h} @ {f:g}"
        self._begin(info)

    def _begin(self, info):
        self.info = info
        self.raw = None
        with self._lock:
            self.frame = None
        w, h = oriented_size(info["w"] or 640, info["h"] or 480, self.w_rot.value)
        if (w, h) != (self.fw, self.fh):
            self.fw, self.fh = w, h
            self.A = self.B = None
            self.set_xy(w // 2, h // 2)
        self.cap_fps = self.disp_fps = 0.0
        self._fails = 0
        try:
            self.read_properties()
        except Exception:
            pass
        self._update_focal()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _halt(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None

    def stop(self):
        self._halt()
        if self.src is not None:
            self.src.close()
        self.src = None
        self._status("stopped (camera released)")

    def close(self):
        self.stop()
        if self.events is not None:
            try:
                self.events.close()
            except Exception:
                pass

    def _settings(self):
        if self.src is None or not self.src.settings_dialog():
            self._status("no settings dialog for this source (DirectShow cameras on Windows only)", err=True)

    # ------------------------------------------------------------- the loop (DD-CP section 11)
    def _loop(self):
        last_disp = last_stats = 0.0
        t_prev = None
        while self._running:
            try:
                if not self.w_freeze.value or self.frame is None:
                    ok, raw = self.src.read()
                    if not ok or raw is None:
                        self._fails += 1
                        if self._fails > 50:
                            self._status("the source stopped delivering frames (unplugged?) - press Start", err=True)
                            self._running = False
                            break
                        time.sleep(0.02)
                        continue
                    self._fails = 0
                    now = time.perf_counter()
                    if t_prev is not None:
                        dt = now - t_prev
                        self.cap_fps = 1.0 / dt if self.cap_fps == 0 else 0.9 * self.cap_fps + 0.1 / max(dt, 1e-6)
                    t_prev = now
                    self._set_frame(raw)
                else:
                    if self._frame_key != self._orient_key() and self.raw is not None:
                        self._set_frame(self.raw)
                    time.sleep(1.0 / max(self.w_dfps.value, 1))
                now = time.perf_counter()
                if now - last_stats >= 0.25:
                    self.step_stats()
                    last_stats = now
                if now - last_disp >= 1.0 / max(self.w_dfps.value, 1):
                    if last_disp:
                        d = now - last_disp
                        self.disp_fps = 1.0 / d if self.disp_fps == 0 else 0.8 * self.disp_fps + 0.2 / d
                    last_disp = now
                    self.step_display()
            except Exception as e:                    # CP-15: report, keep going
                self._status(f"{e.__class__.__name__}: {e}", err=True)
                time.sleep(0.2)

    # ------------------------------------------------------------- per-frame work
    def _sample(self):
        """Points (corner coords) and their BGR values for the current mode."""
        f = self.frame
        x, y = clamp_xy(self.x, self.y, f.shape[1], f.shape[0])
        if self.w_samp.value == "Pin-point":
            pts = np.array([[x, y]])
        else:
            pts = sample_points(f.shape[1], f.shape[0], x, y, self._offsets)
        return pts, f[pts[:, 1], pts[:, 0]]

    def step_stats(self, force=False):
        if self.frame is None:
            return
        with self._lock:
            pts, bgr = self._sample()
        R = self._wheel_r()
        if self.w_samp.value == "Pin-point":
            d = describe_pixel(bgr[0])
            hsv = np.array([d["hsv_cv"]])
            text = pin_text(self.x, self.y, self.fw, self.fh, d)
            st = {"n": 1, "pixel": d}
            sw = self._swatch_html(tuple(int(v) for v in bgr[0]), f"{d['hex']}  rgb{d['rgb']}  hsv{d['hsv_cv']}")
            wheel = render_wheel(hsv, bgr, None, None, R=R)
            table = ""
            keep = np.ones(1, bool)
        else:
            st = cluster_stats(bgr, self.w_exclude.value, self.w_k.value)
            text = cluster_text(self.x, self.y, self.fw, self.fh, self.w_diam.value, pts, st,
                                self.w_exclude.value, self.w_k.value)
            if st["n"]:
                r = st["rows"]
                mh = st["mean_hsv"]
                sw = self._swatch_html(st["mean_bgr"], f"mean of {st['kept']} pts  hsv({mh[0]:.0f}, {mh[1]:.0f}, {mh[2]:.0f})")
                wheel = render_wheel(st["hsv"], bgr, st["keep"], st["mean_hsv"], (r["H"]["min"], r["H"]["max"]),
                                     (r["S"]["min"], r["S"]["max"]), (r["V"]["min"], r["V"]["max"]), R=R)
                keep = st["keep"]
            else:
                sw, wheel, keep = self._swatch_html((128, 128, 128), "—"), render_wheel(R=R), None
            table = stats_html(st)
        self._last_stats, self._last_pts, self._last_keep = st, pts, keep
        self._last_text = text
        if not self.w_hold.value or force:
            if self.w_text.value != text:
                self.w_text.value = text
            self.w_swatch.value = sw
            self.w_wheel.value = encode(wheel, ".png")
            if self.w_table.value != table:
                self.w_table.value = table

    def annotate(self, img, scale=1.0):
        """Draw the overlay on img (a copy of the turned frame scaled by `scale`)."""
        x, y = int(round(self.x * scale)), int(round(self.y * scale))
        col = auto_color(img, x, y) if self.w_color.value == "auto" else COLORS[self.w_color.value]
        t = self.w_thick.value
        if self.A or self.B:
            draw_measure(img, self.A, self.B, scale)
        if self.w_samp.value == "Pin-point":
            draw_crosshair(img, x, y, self.w_gap.value, col, t, self.w_outline.value)
        else:
            r = max(int(round(self.w_diam.value / 2 * scale)), 2)
            pts, keep = self._last_pts, self._last_keep
            if self.w_showpts.value and pts is not None and len(pts) > 1:
                keep = keep if keep is not None and len(keep) == len(pts) else None
                p = np.rint(pts * scale).astype(int)
                for i, (px, py) in enumerate(p[:1500]):
                    c = (0, 0, 255) if keep is not None and not keep[i] else col
                    if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                        img[py, px] = c
            draw_crosshair(img, x, y, r + t + 1, col, t, self.w_outline.value, circle_r=r)
        if self.w_hud.value:
            xc, yc = corner_to_center(self.x, self.y, self.fw, self.fh)
            txt = f"({self.x},{self.y}) c({xc},{yc})  {self.fw}x{self.fh}  {self.cap_fps:.1f} fps"
            if self._orient_key() != (0, False, False):
                txt += f"  rot {self.w_rot.value}" + (" fH" if self.w_fliph.value else "") + (" fV" if self.w_flipv.value else "")
            if self.w_freeze.value:
                txt += "  FROZEN"
            fs = 0.5 if img.shape[1] >= 480 else 0.35
            cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def step_display(self):
        if self.frame is None:
            return
        with self._lock:
            f = self.frame
        s = self._scale()
        if s != 1:
            disp = cv2.resize(f, (max(int(round(self.fw * s)), 1), max(int(round(self.fh * s)), 1)),
                              interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        else:
            disp = f.copy()
        self.w_video.value = encode(self.annotate(disp, s))
        e = self.entry.label if self.entry else ""
        i = self.info
        frozen = " | <b>FROZEN</b>" if self.w_freeze.value else ""
        self.w_status.value = (f"{e} | requested: {self.requested} | driver: {i.get('fmt') or '?'} "
                               f"{i.get('w')}x{i.get('h')} @ {i.get('fps')} | measured: capture "
                               f"{self.cap_fps:.1f} fps, display {self.disp_fps:.1f} fps | modes: "
                               f"{self.modes_from}{frozen}")

    # ------------------------------------------------------------- capture (CP-13)
    def capture(self) -> dict:
        if self.frame is None:
            raise RuntimeError("no frame yet - press Start")
        with self._lock:
            f = self.frame.copy()
            raw_shape = self.raw.shape
        self.step_stats()
        d = Path(self.w_dir.value).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_" + (self.entry.short if self.entry else "src")
        files = {"raw": d / f"{stem}.png"}
        cv2.imwrite(str(files["raw"]), f)
        if self.w_annot.value:
            files["annotated"] = d / f"{stem}_annotated.png"
            cv2.imwrite(str(files["annotated"]), self.annotate(f.copy(), 1.0))
        st = self._last_stats or {}
        stats = None
        if "rows" in st:
            stats = {"n": st["n"], "kept": st["kept"], "rows": st["rows"], "mean_bgr": st["mean_bgr"],
                     "inrange": st["inrange"], "wraps": st["wraps"], "low_sat": st["low_sat"]}
        elif "pixel" in st:
            stats = st["pixel"]
        e = self.entry
        rot, fh_, fv_ = self._orient_key()
        cam_xy = unorient_point(self.x, self.y, raw_shape[1], raw_shape[0], rot, fh_, fv_)
        meta = {"time": datetime.now().isoformat(timespec="milliseconds"), "tool": f"camera_color_picker {_version()}",
                "source": {"label": e.label, "kind": e.kind, "name": e.name, "index": e.index,
                           "vid": e.vid, "pid": e.pid} if e else None,
                "requested": self.requested, "driver": self.info,
                "measured_capture_fps": round(self.cap_fps, 2), "frozen": bool(self.w_freeze.value),
                "orientation": {"rotate_cw_deg": rot, "flip_h": fh_, "flip_v": fv_,
                                "order": "flips first, then rotation", "camera_size": [raw_shape[1], raw_shape[0]],
                                "image_size": [self.fw, self.fh]},
                "crosshair": {"corner": [self.x, self.y],
                              "center": list(corner_to_center(self.x, self.y, self.fw, self.fh)),
                              "camera_pixels": list(cam_xy)},
                "sampling": {"mode": self.w_samp.value, "diameter": self.w_diam.value, "points": self.w_npts.value,
                             "pattern": self.w_pattern.value, "seed": self._seed, "exclude": self.w_exclude.value,
                             "k": self.w_k.value},
                "camera_properties": self.props, "focal": self.focal(),
                "readout": self._last_text, "stats": stats,
                "files": {k: v.name for k, v in files.items()}}
        files["json"] = d / f"{stem}.json"
        files["json"].write_text(json.dumps(meta, indent=2, default=_json_default))
        self.w_saved.value = f"saved <code>{files['raw']}</code>" + (" (+ annotated)" if "annotated" in files else "") + " + .json"
        return {k: str(v) for k, v in files.items()}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
