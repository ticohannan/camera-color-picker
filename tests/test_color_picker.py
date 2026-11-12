"""DD-CP v0.2: the camera color picker, without a camera.

The math and drawing are tested directly; the widget is driven with the built-in test pattern and fake
mouse events; the notebook is executed headless. Needs numpy, opencv; the widget tests need ipywidgets.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from camera_color_picker import analysis, camera, drawing, readout  # noqa: E402


class _All:
    """One namespace over the modules, so the tests read cp.<name> as before the split."""
    def __getattr__(self, name):
        for m in (analysis, camera, drawing, readout):
            if hasattr(m, name):
                return getattr(m, name)
        if name in ("ColorPicker", "_ACTIVE"):
            from camera_color_picker import widget
            return getattr(widget, name)
        raise AttributeError(name)


cp = _All()

FF_OLD = r"""
[dshow @ 000001b2] DirectShow video device options (from video devices)
[dshow @ 000001b2]  Pin "Capture" (alternative pin name "0")
[dshow @ 000001b2]   pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 000001b2]   pixel_format=yuyv422  min s=1280x720 fps=5 max s=1280x720 fps=10
[dshow @ 000001b2]   vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30
"""
FF_NEW = r"""
[dshow @ 0000024d5c] DirectShow video device options (from video devices)
[dshow @ 0000024d5c]  Pin "Capture" (alternative pin name "0")
[dshow @ 0000024d5c]   vcodec=mjpeg  min s=1920x1080 fps=30 max s=1920x1080 fps=30
[dshow @ 0000024d5c]   vcodec=mjpeg  min s=1920x1080 fps=30 max s=1920x1080 fps=30 (tv, bt470bg/bt709/unknown, topleft)
[dshow @ 0000024d5c]   vcodec=mjpeg  min s=640x480 fps=15 max s=640x480 fps=30.0003
[dshow @ 0000024d5c]   vcodec=h264  min s=1920x1080 fps=30 max s=1920x1080 fps=30
[dshow @ 0000024d5c]   pixel_format=yuyv422  min s=640x480 fps=30 max s=640x480 fps=30
video=USB2.0 PC CAMERA: Immediate exit requested
"""


def _hsv_px(h, s, v):
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


# ---------------------------------------------------------------- CP-1, CP-2
def test_sources_listed():
    src = cp.list_sources(include_cameras=False)
    assert [s.kind for s in src] == ["test", "file"]
    e = cp.SourceEntry("camera", 1, "USB2.0 PC CAMERA", 0x058F, 0x5608)
    assert e.label == "1: USB2.0 PC CAMERA (058F:5608)"
    assert e.short == "cam1-USB2-0-PC-CAMERA"


def test_parse_dshow_options():
    m = cp.parse_dshow_options(FF_OLD)
    assert [(x.fmt, x.w, x.h, x.fps_min, x.fps_max) for x in m] == [
        ("MJPG", 1920, 1080, 5, 30), ("YUY2", 1280, 720, 5, 10), ("YUY2", 640, 480, 5, 30)]
    m = cp.parse_dshow_options(FF_NEW)          # duplicates merged, H264 dropped, fractional fps
    assert [(x.fmt, x.w, x.h) for x in m] == [("MJPG", 1920, 1080), ("MJPG", 640, 480), ("YUY2", 640, 480)]
    assert abs(m[1].fps_max - 30.0003) < 1e-6 and m[1].fps_min == 15
    assert m[0].label == "MJPG 1920x1080 30 fps" and m[1].label.startswith("MJPG 640x480 15-30")
    assert cp.parse_dshow_options("nothing useful") == []


def test_fps_choices():
    assert cp.fps_choices(cp.Mode("MJPG", 640, 480, 5, 30)) == [5, 7.5, 10, 15, 20, 24, 25, 30]
    assert cp.fps_choices(cp.Mode("YUY2", 1280, 720, 5, 10)) == [5, 7.5, 10]
    assert cp.fps_choices(cp.Mode("", 640, 480, 0, 0)) == [5, 7.5, 10, 15, 20, 24, 25, 30]
    assert cp.fps_choices(cp.Mode("", 3264, 2448, 1, 2.5)) == [2.5]


# ---------------------------------------------------------------- CP-6
def test_coords_roundtrip():
    assert cp.corner_to_center(320, 240, 640, 480) == (0, 0)
    assert cp.corner_to_center(0, 0, 640, 480) == (-320, -240)
    assert cp.corner_to_center(639, 479, 640, 480) == (319, 239)
    assert cp.corner_to_center(0, 0, 641, 481) == (-320, -240)
    for x, y in [(0, 0), (17, 400), (639, 479)]:
        assert cp.center_to_corner(*cp.corner_to_center(x, y, 640, 480), 640, 480) == (x, y)
    assert cp.clamp_xy(-5, 900, 640, 480) == (0, 479)


# ---------------------------------------------------------------- CP-7
def test_overlay_gap_and_circle():
    img = np.zeros((101, 101, 3), np.uint8)
    cp.draw_crosshair(img, 50, 50, 6, (255, 255, 255), 1)
    assert img[50, 45:56].sum() == 0 and img[45:56, 50].sum() == 0               # gap kept clear (AA edge at +-6)
    assert img[50, 58].sum() > 0 and img[50, 30].sum() > 0 and img[20, 50].sum() > 0 and img[90, 50].sum() > 0
    img = np.zeros((101, 101, 3), np.uint8)
    cp.draw_crosshair(img, 50, 50, 22, (255, 255, 255), 1, circle_r=20)
    assert img[50, 70].sum() > 0 and img[30, 50].sum() > 0 and img[50, 60].sum() == 0    # circle, empty inside
    img = np.zeros((101, 101, 3), np.uint8)
    cp.draw_crosshair(img, 2, 2, 6, (255, 255, 255), 1)                                  # near a corner: no error
    assert cp.auto_color(np.full((50, 50, 3), 250, np.uint8), 25, 25) == (0, 0, 0)
    assert cp.auto_color(np.full((50, 50, 3), 10, np.uint8), 25, 25) == (255, 255, 255)


# ---------------------------------------------------------------- CP-8
def test_pin_readout_known_pixel():
    d = cp.describe_pixel(np.array([40, 34, 201]))            # BGR
    assert d["rgb"] == (201, 34, 40) and d["hex"] == "#C92228"
    h, s, v = d["hsv_cv"]
    assert h == 179 and s == 212 and v == 201
    assert d["hsv_deg_pct"][0] == h * 2.0 and abs(d["hsv_matlab"][1] - 212 / 255) < 1e-3
    t = cp.pin_text(400, 100, 640, 480, d)
    assert "xy_corner=(400, 100)" in t and "xy_center=(80, -140)" in t and "hex=#C92228" in t


# ---------------------------------------------------------------- CP-9
def test_disk_offsets():
    for pat in ("random", "even"):
        o = cp.disk_offsets(500, 41, pat, seed=3)
        assert tuple(o[0]) == (0, 0)
        assert ((o ** 2).sum(1) <= 20.5 ** 2).all()
        assert len(np.unique(o, axis=0)) == len(o) and len(o) > 200
        assert abs(o[:, 0].mean()) < 2 and abs(o[:, 1].mean()) < 2
    a, b = cp.disk_offsets(100, 41, "random", 1), cp.disk_offsets(100, 41, "random", 2)
    assert not np.array_equal(a, b)                            # Re-scatter gives a new draw
    assert np.array_equal(a, cp.disk_offsets(100, 41, "random", 1))
    p = cp.sample_points(640, 480, 2, 2, cp.disk_offsets(300, 41))
    assert (p >= 0).all() and len(p) < 300                    # points off the frame dropped


# ---------------------------------------------------------------- CP-10, CP-11
def test_circular_hue_stats_wrap():
    m, s, R = cp.hue_circular([176, 178, 2, 4])
    assert min(abs(m - 180), abs(m)) < 0.01 and R > 0.98      # red, not cyan (90)
    assert np.mean([176, 178, 2, 4]) == 90                    # the naive mean would be cyan
    m, s, R = cp.hue_circular([60] * 10)
    assert abs(m - 60) < 1e-9 and s < 1e-6 and abs(R - 1) < 1e-12
    assert list(cp.hue_diff([178, 2, 90], 0)) == [-2, 2, -90]


def test_hue_arc_and_inrange_split():
    assert cp.hue_arc([10, 20, 15]) == (10, 20)
    assert cp.hue_arc([175, 178, 3, 5]) == (175, 5)
    assert cp.hue_arc([42]) == (42, 42)
    assert cp.inrange_pairs(10, 20, 100, 200, 50, 250) == [((10, 100, 50), (20, 200, 250))]
    assert cp.inrange_pairs(175, 5, 100, 200, 50, 250) == [((175, 100, 50), (179, 200, 250)),
                                                         ((0, 100, 50), (5, 200, 250))]


def test_cluster_on_two_colors():
    red_a = [_hsv_px(176, 210, 200)] * 50
    red_b = [_hsv_px(3, 210, 200)] * 50
    st = cp.cluster_stats(np.array(red_a + red_b))
    r = st["rows"]
    assert st["n"] == 100 and st["kept"] == 100 and st["wraps"]
    assert (r["H"]["min"], r["H"]["max"]) == (176, 3)
    assert min(r["H"]["mean"], 180 - r["H"]["mean"]) < 3.0
    assert len(st["inrange"]) == 2 and st["inrange"][0][1][0] == 179 and st["inrange"][1][0][0] == 0
    # the masks from these ranges really do select every sample
    img = np.array(red_a + red_b, np.uint8).reshape(10, 10, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in st["inrange"])
    assert (mask > 0).all()
    t = cp.cluster_text(100, 100, 640, 480, 41, np.array([[90, 95], [110, 104]]), st, False, 2.0)
    assert "lower2" in t and "hue wraps" in t
    grey = cp.cluster_stats(np.full((20, 3), 128, np.uint8))
    assert grey["low_sat"]


def test_outlier_exclusion():
    rng = np.random.default_rng(0)
    good = [_hsv_px(int(h), 200, 180) for h in rng.integers(58, 63, 95)]
    bad = [_hsv_px(120, 200, 180)] * 5                         # 5 blue points among green
    st = cp.cluster_stats(np.array(good + bad), exclude=False)
    assert st["kept"] == 100 and st["rows"]["H"]["max"] == 120
    st = cp.cluster_stats(np.array(good + bad), exclude=True, k=2.0)
    assert st["kept"] == 95 and not st["keep"][-5:].any()
    assert st["rows"]["H"]["max"] <= 63
    flat = cp.cluster_stats(np.full((30, 3), 77, np.uint8), exclude=True)   # sd = 0: nothing excluded
    assert flat["kept"] == 30


# ---------------------------------------------------------------- CP-12
def test_wheel_point_positions():
    R, pad = cp.WHEEL_R, 26
    c = R + pad
    assert cp.wheel_xy(0, 255) == (c + R, c)              # red at the right
    assert cp.wheel_xy(45, 255) == (c, c - R)             # 90 deg (chartreuse) at the top
    assert cp.wheel_xy(90, 255) == (c - R, c)             # cyan at the left
    assert cp.wheel_xy(30, 0) == (c, c)
    base = cp.render_wheel()
    # the disc's own colour at a sample's position matches the sample's hue
    for h in (0, 30, 60, 120):
        x, y = cp.wheel_xy(h, 230)
        hh = int(cv2.cvtColor(base[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0, 0])
        assert min(abs(hh - h), 180 - abs(hh - h)) <= 2
    hsv = np.array([[60, 200, 150], [120, 200, 150]])
    bgr = np.array([_hsv_px(*p) for p in hsv])
    img = cp.render_wheel(hsv, bgr, np.array([True, False]), (60, 200, 150), (60, 60), (200, 200), (150, 150))
    x, y = cp.wheel_xy(120, 200)
    ring = [img[y - 4, x], img[y + 4, x]]
    assert any(p[2] > 200 and p[0] < 80 and p[1] < 80 for p in ring)        # excluded: ringed red
    assert img.shape == base.shape


# ---------------------------------------------------------------- widget (needs ipywidgets)
def _widgets():
    try:
        import ipywidgets  # noqa: F401
        return True
    except ImportError:
        print("  (ipywidgets not installed: skipped)")
        return False


def _picker(tmp):
    p = cp.ColorPicker(save_dir=tmp, list_cams=False)
    p.w_src.value = 0                                          # test pattern
    p.start()
    t0 = time.time()
    while p.frame is None and time.time() - t0 < 5:
        time.sleep(0.02)
    assert p.frame is not None
    return p


def test_ui_mouse_and_go():
    if not _widgets():
        return
    with tempfile.TemporaryDirectory() as tmp:
        p = _picker(tmp)
        try:
            assert (p.x, p.y) == (320, 240)
            p.w_vwidth.value = 800                             # display is 1.25 x the frame
            p._on_mouse({"type": "mousedown", "dataX": 150, "dataY": 150, "buttons": 1})
            assert (p.x, p.y) == (120, 120)                   # inside the red patch
            p._on_mouse({"type": "mousemove", "dataX": 400, "dataY": 400, "buttons": 0})
            assert (p.x, p.y) == (120, 120)                   # moving without a button does nothing
            p._on_mouse({"type": "mousemove", "dataX": 10000, "dataY": -5, "buttons": 1})
            assert (p.x, p.y) == (639, 0)                     # clamped
            p.w_origin.value = "Center"
            assert (p.w_x.value, p.w_y.value) == (319, -240)
            p.w_x.value, p.w_y.value = -200, -120
            p._go()
            assert (p.x, p.y) == (120, 120)
            p.w_origin.value = "Corner"
            p.w_x.value, p.w_y.value = 520, 160
            p._go()
            assert (p.x, p.y) == (520, 160)                   # green patch
            p.w_center.click()
            assert (p.x, p.y) == (320, 240)
            p.set_xy(520, 160)
            p.step_stats()
            assert "mode=pin" in p.w_text.value and "xy_center=(200, -80)" in p.w_text.value
            p.w_samp.value = "Cluster"
            p.set_xy(120, 120)
            p.w_exclude.value = True
            p.step_stats()
            st = p._last_stats
            assert st["kept"] > 0.9 * st["n"] and st["wraps"]                       # the red patch straddles 0/180
            assert "lower2" in p.w_text.value and "<table" in p.w_table.value
            time.sleep(0.3)
            assert p.w_video.value[:2] == b"\xff\xd8"           # JPEG frames reach the widget
            assert p.cap_fps > 5
        finally:
            p.close()


def test_capture_files():
    if not _widgets():
        return
    with tempfile.TemporaryDirectory() as tmp:
        p = _picker(tmp)
        try:
            p.w_samp.value = "Cluster"
            p.set_xy(320, 100)                                # woodgrain
            p.step_stats()
            files = p.capture()
            raw = cv2.imread(files["raw"])
            assert raw.shape == (480, 640, 3)
            assert abs(int(raw[100, 320].astype(int).sum()) - int(cp.make_test_pattern()[100, 320].astype(int).sum())) < 40
            ann = cv2.imread(files["annotated"])
            assert ann.shape == raw.shape and (ann != raw).any()
            meta = json.loads(Path(files["json"]).read_text())
            assert meta["crosshair"] == {"corner": [320, 100], "center": [0, -140], "camera_pixels": [320, 100]}
            assert meta["source"]["kind"] == "test" and meta["stats"]["n"] > 100
            assert Path(files["raw"]).name.endswith("_testpattern.png")
        finally:
            p.close()


def test_ui_freeze_hold_and_rerun():
    if not _widgets():
        return
    with tempfile.TemporaryDirectory() as tmp:
        p = _picker(tmp)
        p.w_freeze.value = True
        time.sleep(0.2)
        f1 = p.frame
        time.sleep(0.3)
        assert p.frame is f1                                   # frozen: no new frames
        p.w_hold.value = True
        p.step_stats()
        before = p.w_text.value
        p.set_xy(10, 10)
        p.step_stats()
        assert p.w_text.value == before                        # held readout
        q = cp.ColorPicker(save_dir=tmp, list_cams=False)      # "running the cell again"
        try:
            assert p.src is None and not p._running            # the old one released its source
            assert cp._ACTIVE == [q]
        finally:
            q.close()


def test_notebook_executes():
    if not _widgets():
        return
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("  (nbclient not installed: skipped)")
        return
    nb = nbformat.read(ROOT / "notebooks" / "color_picker.ipynb", as_version=4)
    NotebookClient(nb, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    out = nb.cells[1].outputs
    assert not [o for o in out if o.output_type == "error"], out
    assert any("application/vnd.jupyter.widget-view+json" in o.get("data", {}) for o in out)


# ---------------------------------------------------------------- v0.2: CP-12 sizes, CP-17 ... CP-20
def test_wheel_sizes():
    for R in (60, 100, 160):
        img = cp.render_wheel(R=R)
        assert img.shape[0] == 2 * (R + 26)
        x, y = cp.wheel_xy(0, 255, R=R)
        assert (x, y) == (2 * R + 26, R + 26)


def test_orient_image_and_points():
    w, h = 7, 4
    img = np.zeros((h, w, 3), np.uint8)
    for rot in (0, 90, 180, 270):
        for fh_ in (False, True):
            for fv_ in (False, True):
                for (x, y) in [(0, 0), (6, 0), (2, 3), (5, 1)]:
                    im = img.copy()
                    im[y, x] = (255, 255, 255)
                    o = cp.orient(im, rot, fh_, fv_)
                    assert (o.shape[1], o.shape[0]) == cp.oriented_size(w, h, rot)
                    ox, oy = cp.orient_point(x, y, w, h, rot, fh_, fv_)
                    assert o[oy, ox, 0] == 255 and o.sum() == 255 * 3          # the image and the map agree
                    assert cp.unorient_point(ox, oy, w, h, rot, fh_, fv_) == (x, y)
    # a camera upside down = 180; a mirror = flip horizontal
    assert cp.orient_point(0, 0, w, h, 180) == (6, 3)
    assert cp.orient_point(0, 2, w, h, 0, True) == (6, 2)
    assert cp.orient_point(0, 0, w, h, 90) == (3, 0)                         # top-left goes to top-right


def test_focal_from_measurement():
    # pinhole: 100 mm object at 500 mm spans 160 px -> f = 800 px
    f = cp.focal_from_measurement(160, 100, 500)
    assert f == 800
    fov = cp.fov_from_focal(800, 640, 480)
    assert abs(fov["h"] - 2 * np.degrees(np.arctan(320 / 800))) < 1e-9
    assert abs(fov["v"] - 2 * np.degrees(np.arctan(240 / 800))) < 1e-9
    assert abs(fov["d"] - 2 * np.degrees(np.arctan(400 / 800))) < 1e-9       # diagonal 800 px
    assert abs(fov["px_per_deg"] - 800 * np.pi / 180) < 1e-9
    assert cp.focal_from_measurement(0, 100, 500) is None and cp.focal_from_measurement(10, 0, 500) is None


def test_focal_from_fov():
    f = cp.focal_from_dfov(55.0, 640, 480)                                  # a C270-class webcam
    assert abs(cp.fov_from_focal(f, 640, 480)["d"] - 55.0) < 1e-9
    assert 760 < f < 770
    assert cp.focal_from_dfov(0, 640, 480) is None and cp.focal_from_dfov(180, 640, 480) is None
    t = cp.focal_text(640, 480, (100, 200), (260, 200), 100, 500, 800.0, 55.0, f, 3.0)
    assert "|AB| = 160.0 px" in t and "f = 800.0 px" in t and "f = 2.40 mm" in t and "f_px=800.0 @ 640x480" in t


class _FakeCap:
    def __init__(self, vals):
        self.vals = vals

    def getBackendName(self):
        return "DSHOW"

    def get(self, pid):
        return self.vals.get(pid, -1.0)


def test_camera_properties():
    p = cp.read_properties(_FakeCap({cv2.CAP_PROP_EXPOSURE: -6.0, cv2.CAP_PROP_FOCUS: 0.0}))
    assert p["backend"] == "DSHOW" and p["exposure"] == -6.0 and p["focus"] == 0.0
    assert p["zoom"] is None and p["gain"] is None
    html = cp.props_html(p)
    assert "not reported" in html and "-6.0" in html


def test_ui_size_and_layout():
    if not _widgets():
        return
    with tempfile.TemporaryDirectory() as tmp:
        p = _picker(tmp)
        try:
            assert p.body.layout.flex_flow == "column"
            p.w_layout.value = "side"
            assert p.body.layout.flex_flow == "row" and len(p.body.children) == 2
            assert p.body.children[0].children[0] is p.w_video
            p.w_layout.value = "stacked"
            assert p.body.children[0] is p.w_video
            p.w_vwidth.value = 320
            assert p.w_video.layout.width == "320px"
            time.sleep(0.3)
            disp = cv2.imdecode(np.frombuffer(p.w_video.value, np.uint8), cv2.IMREAD_COLOR)
            assert disp.shape[1] == 320                                      # encoded at the chosen width
            p._on_mouse({"type": "mousedown", "dataX": 60, "dataY": 60, "buttons": 1})
            assert (p.x, p.y) == (120, 120)                                   # still full-resolution pixels
            p.w_fit.value = True
            assert p.w_video.layout.width == "100%"
            p.w_wsize.value = 160
            wheel = cv2.imdecode(np.frombuffer(p.w_wheel.value, np.uint8), cv2.IMREAD_COLOR)
            assert wheel.shape[0] == 2 * (80 + 26)
        finally:
            p.close()


def test_ui_orientation():
    if not _widgets():
        return
    with tempfile.TemporaryDirectory() as tmp:
        p = _picker(tmp)
        try:
            p.set_xy(120, 120)                                                # red patch
            p.w_rot.value = 90
            time.sleep(0.2)
            assert (p.fw, p.fh) == (480, 640) and p.frame.shape[:2] == (640, 480)
            assert (p.x, p.y) == (240, 320)                                   # re-centered after the size change
            x, y = cp.orient_point(120, 120, 640, 480, 90)
            p.set_xy(x, y)
            p.step_stats()
            assert p._last_stats["pixel"]["hsv_cv"][0] in range(170, 180) or p._last_stats["pixel"]["hsv_cv"][0] < 10
            files = p.capture()
            meta = json.loads(Path(files["json"]).read_text())
            assert meta["orientation"]["rotate_cw_deg"] == 90 and meta["orientation"]["image_size"] == [480, 640]
            assert meta["crosshair"]["camera_pixels"] == [120, 120]
            assert cv2.imread(files["raw"]).shape[:2] == (640, 480)
            p.w_rot.value = 0
            p.w_fliph.value = True
            p.w_freeze.value = True                                           # a frozen frame turns too
            time.sleep(0.2)
            f1 = p.frame.copy()
            p.w_flipv.value = True
            time.sleep(0.3)
            assert np.array_equal(p.frame, cv2.flip(f1, 0))
        finally:
            p.close()


def test_ui_measure_focal():
    if not _widgets():
        return
    with tempfile.TemporaryDirectory() as tmp:
        p = _picker(tmp)
        try:
            p.set_xy(240, 120)
            p.w_setA.click()
            p.set_xy(400, 120)
            p.w_setB.click()
            p.w_len.value, p.w_dist.value = 100.0, 500.0
            f = p.focal()
            assert f["ab_px"] == 160 and f["f_px_measured"] == 800
            assert "f = 800.0 px" in p.w_focal.value
            p.w_dfov.value = 55.0
            assert p.focal()["f_px_datasheet"] is not None and "datasheet" in p.w_focal.value
            assert p.props.get("backend") == "test pattern"
            files = p.capture()
            meta = json.loads(Path(files["json"]).read_text())
            assert meta["focal"]["f_px_measured"] == 800 and meta["camera_properties"]["backend"] == "test pattern"
            p.w_clearAB.click()
            assert p.A is None and "Set A" in p.w_focal.value
        finally:
            p.close()
