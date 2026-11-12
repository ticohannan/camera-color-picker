"""Video sources: DirectShow cameras (Windows), a test pattern and still image files (DD-CP sections 4, 8)."""
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

IS_WINDOWS = sys.platform == "win32"
BACKEND = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY


@dataclass
class SourceEntry:
    kind: str                     # "camera" | "test" | "file"
    index: int = -1               # DirectShow index for cameras
    name: str = ""
    vid: Optional[int] = None
    pid: Optional[int] = None
    path: str = ""
    dup_number: int = 0           # n-th camera with this same name (FFmpeg -video_device_number)

    @property
    def label(self) -> str:
        if self.kind == "test":
            return "Test pattern (no camera)"
        if self.kind == "file":
            return "Still image file (path below)"
        ids = f" ({self.vid:04X}:{self.pid:04X})" if self.vid is not None and self.pid is not None else ""
        return f"{self.index}: {self.name}{ids}"

    @property
    def short(self) -> str:
        """Letters and digits only, for file names."""
        base = {"test": "testpattern", "file": "file"}.get(self.kind, f"cam{self.index}_{self.name}")
        return re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-")[:40]


class _quiet_opencv:
    """Silence OpenCV's own log while probing indices that may not exist."""
    def __enter__(self):
        try:
            self.old = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            self.old = None

    def __exit__(self, *a):
        if self.old is not None:
            cv2.utils.logging.setLogLevel(self.old)


def list_cameras(probe_max: int = 6) -> List[SourceEntry]:
    """Cameras in the order the capture backend numbers them (DirectShow on Windows). CP-1, DD-CP section 4."""
    cams: List[SourceEntry] = []
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        api = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
        for c in enumerate_cameras(api):
            cams.append(SourceEntry("camera", int(c.index) % 100, c.name or f"Camera {c.index}",
                                    c.vid, c.pid, c.path or ""))
    except Exception:
        cams = []
    if not cams and IS_WINDOWS:
        try:
            from pygrabber.dshow_graph import FilterGraph
            for i, n in enumerate(FilterGraph().get_input_devices()):
                cams.append(SourceEntry("camera", i, n))
        except Exception:
            cams = []
    if not cams:
        with _quiet_opencv():
            for i in range(probe_max):
                cap = cv2.VideoCapture(i, BACKEND)
                ok = cap.isOpened()
                cap.release()
                if ok:
                    cams.append(SourceEntry("camera", i, f"Camera {i}"))
    seen = {}
    for c in cams:
        c.dup_number = seen.get(c.name, 0)
        seen[c.name] = c.dup_number + 1
    return cams


def list_sources(include_cameras: bool = True) -> List[SourceEntry]:
    out = list_cameras() if include_cameras else []
    return out + [SourceEntry("test", name="test pattern"), SourceEntry("file", name="image file")]


@dataclass
class Mode:
    fmt: str            # FOURCC as 4 characters ("MJPG", "YUY2"), "" = driver default
    w: int
    h: int
    fps_min: float
    fps_max: float

    @property
    def label(self) -> str:
        f = self.fmt or "default"
        if self.fps_min and abs(self.fps_min - self.fps_max) > 0.01:
            r = f"{self.fps_min:g}-{self.fps_max:g} fps"
        else:
            r = f"{self.fps_max:g} fps" if self.fps_max else "fps ?"
        return f"{f} {self.w}x{self.h} {r}"


FFMPEG_FMT = {"mjpeg": "MJPG", "yuyv422": "YUY2", "nv12": "NV12", "h264": "H264", "bgr24": "BGR3",
              "rgb24": "RGB3", "yuv420p": "I420", "uyvy422": "UYVY", "gray": "GREY"}


_DSHOW_RE = re.compile(r"(vcodec|pixel_format)=(\S+)\s+min s=(\d+)x(\d+) fps=([\d.]+)\s+max s=(\d+)x(\d+) fps=([\d.]+)")


COMMON_SIZES = [(160, 120), (320, 240), (352, 288), (640, 360), (640, 480), (800, 600), (960, 720),
                (1024, 768), (1280, 720), (1280, 960), (1280, 1024), (1600, 1200), (1920, 1080),
                (2048, 1536), (2592, 1944), (3264, 2448)]


STD_FPS = [5, 7.5, 10, 15, 20, 24, 25, 30, 50, 60, 90, 120]


def parse_dshow_options(text: str) -> List[Mode]:
    """Modes from `ffmpeg -f dshow -list_options true` output (stderr). Duplicates removed; formats OpenCV
    cannot use (H264) dropped; sorted by format, then largest first."""
    modes = {}
    for m in _DSHOW_RE.finditer(text):
        kind, val, w1, h1, f1, w2, h2, f2 = m.groups()
        fmt = FFMPEG_FMT.get(val, val.upper()[:4])
        if fmt == "H264":
            continue
        w, h = int(w2), int(h2)
        lo, hi = float(f1), float(f2)
        key = (fmt, w, h)
        if key in modes:
            o = modes[key]
            modes[key] = Mode(fmt, w, h, min(o.fps_min, lo), max(o.fps_max, hi))
        else:
            modes[key] = Mode(fmt, w, h, lo, hi)
    order = {"MJPG": 0, "YUY2": 1}
    return sorted(modes.values(), key=lambda m: (order.get(m.fmt, 2), m.fmt, -m.w * m.h, -m.w))


def find_ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def list_modes_ffmpeg(entry: SourceEntry, timeout: float = 20.0) -> Optional[List[Mode]]:
    """None when FFmpeg is not available or not on Windows; [] when it ran but found nothing."""
    exe = find_ffmpeg()
    if not exe or not IS_WINDOWS or entry.kind != "camera":
        return None
    cmd = [exe, "-hide_banner", "-f", "dshow", "-list_options", "true",
           "-video_device_number", str(entry.dup_number), "-i", f"video={entry.name}"]
    try:
        flags = 0x08000000 if IS_WINDOWS else 0          # CREATE_NO_WINDOW
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=flags)
    except Exception:
        return None
    return parse_dshow_options(p.stderr + p.stdout)


def fourcc_str(v: float) -> str:
    i = int(v)
    s = "".join(chr((i >> 8 * k) & 0xFF) for k in range(4))
    return s if s.isprintable() and s.strip() else ""


def probe_modes(cap) -> List[Mode]:
    """Fallback (no FFmpeg): ask the driver for each common size and keep what it reports back."""
    found = {}
    with _quiet_opencv():
        _probe(cap, found)
    return sorted(found.values(), key=lambda m: (m.fmt != "MJPG", -m.w * m.h))


def _probe(cap, found):
    for fmt in ("", "MJPG"):
        if fmt:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
            if fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) != fmt:
                continue
        for w, h in COMMON_SIZES:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            gw, gh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if gw > 0 and gh > 0:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                found[(fmt, gw, gh)] = Mode(fmt, gw, gh, 0.0, float(round(fps, 2)))


def fps_choices(mode: Mode) -> List[float]:
    lo = mode.fps_min or 1.0
    hi = mode.fps_max or 30.0
    out = [f for f in STD_FPS if lo - 0.01 <= f <= hi + 0.01]
    if not out or abs(out[-1] - hi) > 0.01:
        out.append(round(hi, 2))
    return out


PROPS = [("focus", "CAP_PROP_FOCUS"), ("autofocus", "CAP_PROP_AUTOFOCUS"), ("zoom", "CAP_PROP_ZOOM"),
         ("exposure", "CAP_PROP_EXPOSURE"), ("auto exposure", "CAP_PROP_AUTO_EXPOSURE"), ("gain", "CAP_PROP_GAIN"),
         ("brightness", "CAP_PROP_BRIGHTNESS"), ("contrast", "CAP_PROP_CONTRAST"),
         ("saturation", "CAP_PROP_SATURATION"), ("white balance (K)", "CAP_PROP_WB_TEMPERATURE"),
         ("auto white balance", "CAP_PROP_AUTO_WB"), ("sharpness", "CAP_PROP_SHARPNESS"),
         ("backlight comp.", "CAP_PROP_BACKLIGHT")]


def read_properties(cap) -> dict:
    """What the driver reports (CP-19). None = not reported (-1, unsupported or the call failed)."""
    out = {}
    try:
        out["backend"] = cap.getBackendName()
    except Exception:
        out["backend"] = None
    for label, attr in PROPS:
        pid = getattr(cv2, attr, None)
        v = None
        if pid is not None:
            try:
                v = float(cap.get(pid))
            except Exception:
                v = None
        out[label] = None if v is None or v == -1 else round(v, 3)
    return out


class CameraSource:
    """A DirectShow (Windows) camera through OpenCV."""

    def __init__(self, entry: SourceEntry):
        self.entry = entry
        self.cap = None
        self.info = {}

    def open(self, mode: Optional[Mode] = None, fps: Optional[float] = None) -> dict:
        self.close()
        cap = cv2.VideoCapture(self.entry.index, BACKEND)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.entry.label} (in use by another program?)")
        if mode:
            for order in ("fourcc_first", "size_first"):
                if order == "fourcc_first" and mode.fmt:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode.fmt))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.h)
                if order == "size_first" and mode.fmt:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode.fmt))
                if not mode.fmt or fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) == mode.fmt:
                    break
        if fps:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        self.cap = cap
        self.info = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     "fps": round(cap.get(cv2.CAP_PROP_FPS), 2), "fmt": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}
        return self.info

    def modes(self) -> Tuple[List[Mode], str]:
        m = list_modes_ffmpeg(self.entry)
        if m:
            return m, "ffmpeg"
        self.open()
        m = probe_modes(self.cap)
        self.close()
        return m, "probed (install imageio-ffmpeg for exact modes)"

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def settings_dialog(self) -> bool:
        return bool(self.cap is not None and self.cap.set(cv2.CAP_PROP_SETTINGS, 1))

    def properties(self) -> dict:
        return read_properties(self.cap) if self.cap is not None else {}

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None


def make_test_pattern(w: int = 640, h: int = 480) -> np.ndarray:
    """A still scene with known colors: red that straddles the hue wrap, a woodgrain patch, a magenta,
    a green and a blue block, a gray card, and a hue strip. Used without a camera and by the tests."""
    img = np.full((h, w, 3), (205, 200, 195), np.uint8)
    s = w / 640.0
    hsv = np.zeros((h // 8, w, 3), np.uint8)
    hsv[..., 0] = (np.arange(w) * 180 // w).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    img[h - h // 8:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def box(x0, y0, x1, y1):
        return slice(int(y0 * s), int(y1 * s)), slice(int(x0 * s), int(x1 * s))

    # red, left half H=176, right half H=3 (crosses 0/180)
    r = box(40, 40, 200, 200)
    sub = img[r]
    hh = np.zeros(sub.shape, np.uint8)
    hh[..., 0] = 176
    hh[:, sub.shape[1] // 2:, 0] = 3
    hh[..., 1] = 210
    hh[..., 2] = 200
    img[r] = cv2.cvtColor(hh, cv2.COLOR_HSV2BGR)
    # woodgrain
    r = box(240, 40, 400, 200)
    yy, xx = np.mgrid[0:img[r].shape[0], 0:img[r].shape[1]]
    g = 0.5 + 0.5 * np.sin(xx / (6 * s) + 2.0 * np.sin(yy / (25 * s)))
    wood = np.stack([40 + 30 * g, 95 + 45 * g, 150 + 50 * g], -1)
    img[r] = np.clip(wood, 0, 255).astype(np.uint8)
    img[box(440, 40, 600, 120)] = (180, 40, 200)        # magenta (BGR)
    img[box(440, 120, 600, 200)] = (60, 170, 40)        # green
    img[box(40, 240, 200, 380)] = (190, 90, 30)         # blue
    img[box(240, 240, 400, 380)] = (128, 128, 128)      # gray card
    img[box(440, 240, 600, 380)] = (40, 200, 230)       # yellow
    return img


class TestPatternSource:
    MODES = [Mode("", 640, 480, 1, 30), Mode("", 1280, 720, 1, 30), Mode("", 320, 240, 1, 60)]

    def __init__(self, entry=None, noise: float = 4.0):
        self.entry = entry or SourceEntry("test", name="test pattern")
        self.noise = noise
        self.base = make_test_pattern()
        self.fps = 30.0
        self._t = 0.0
        self.info = {}
        self._rng = np.random.default_rng(1)

    def open(self, mode=None, fps=None):
        m = mode or self.MODES[0]
        self.base = make_test_pattern(m.w, m.h)
        self.fps = float(fps or m.fps_max)
        self.info = {"w": m.w, "h": m.h, "fps": self.fps, "fmt": "TEST"}
        return self.info

    def modes(self):
        return list(self.MODES), "built-in"

    def read(self):
        wait = 1.0 / self.fps - (time.perf_counter() - self._t)
        if wait > 0:
            time.sleep(wait)
        self._t = time.perf_counter()
        if self.noise:
            n = self._rng.normal(0, self.noise, self.base.shape)
            return True, np.clip(self.base + n, 0, 255).astype(np.uint8)
        return True, self.base.copy()

    def settings_dialog(self):
        return False

    def properties(self):
        return {"backend": "test pattern" if self.entry.kind == "test" else "image file"}

    def close(self):
        pass


class FileSource(TestPatternSource):
    def __init__(self, entry, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read image file {path}")
        super().__init__(entry, noise=0.0)
        self.img = img
        self.fps = 10.0

    def open(self, mode=None, fps=None):
        self.base = self.img
        self.fps = float(fps or 10.0)
        self.info = {"w": self.img.shape[1], "h": self.img.shape[0], "fps": self.fps, "fmt": "FILE"}
        return self.info

    def modes(self):
        return [Mode("", self.img.shape[1], self.img.shape[0], 1, 10)], "file"
