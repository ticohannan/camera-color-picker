"""Overlay, color wheel and image encoding (DD-CP sections 6, 9)."""
import math

import cv2
import numpy as np


COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (0, 0, 255), "green": (0, 200, 0),
          "blue": (255, 80, 0), "yellow": (0, 255, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255)}


def auto_color(img, x, y, half=30):
    h, w = img.shape[:2]
    patch = img[max(0, y - half):min(h, y + half + 1), max(0, x - half):min(w, x + half + 1)]
    lum = float((patch.reshape(-1, 3) @ np.array([0.114, 0.587, 0.299])).mean()) if patch.size else 0.0
    return (0, 0, 0) if lum > 128 else (255, 255, 255)


def draw_crosshair(img, x, y, gap, color, thick=1, outline=False, circle_r=None):
    """Crosshair through (x, y) leaving a gap of `gap` px each side of the center; optional circle."""
    h, w = img.shape[:2]
    shade = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    segs = [((0, y), (x - gap - 1, y)), ((x + gap + 1, y), (w - 1, y)),
            ((x, 0), (x, y - gap - 1)), ((x, y + gap + 1), (x, h - 1))]
    passes = ([(shade, thick + 2)] if outline else []) + [(color, thick)]
    for col, t in passes:
        for a, b in segs:
            if b[0] >= a[0] and b[1] >= a[1]:          # skip a segment squeezed out by the gap
                cv2.line(img, a, b, col, t, cv2.LINE_AA)
        if circle_r:
            cv2.circle(img, (x, y), int(circle_r), col, t, cv2.LINE_AA)
    return img


WHEEL_R = 130


_WHEEL_CACHE = {}


def _wheel_base(R=WHEEL_R, pad=26, bar=34):
    key = (R, pad, bar)
    if key in _WHEEL_CACHE:
        return _WHEEL_CACHE[key].copy()
    size = 2 * (R + pad)
    img = np.full((size, size + bar + 30, 3), 40, np.uint8)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    dx, dy = xx - (R + pad), (R + pad) - yy
    rr = np.hypot(dx, dy)
    hue = (np.degrees(np.arctan2(dy, dx)) % 360) / 2.0         # Python atan2(y, x) = class Atan2(x, y)
    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[..., 0] = np.clip(hue, 0, 179.999).astype(np.uint8)
    hsv[..., 1] = np.clip(rr / R * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    disc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    inside = rr <= R
    img[:, :size][inside] = disc[inside]
    c = R + pad
    cv2.circle(img, (c, c), R, (90, 90, 90), 1, cv2.LINE_AA)
    for deg, lab in zip(range(0, 360, 60), "RYGCBM"):
        a = math.radians(deg)
        p1 = (int(c + R * math.cos(a)), int(c - R * math.sin(a)))
        p2 = (int(c + (R + 6) * math.cos(a)), int(c - (R + 6) * math.sin(a)))
        cv2.line(img, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        tp = (int(c + (R + 15) * math.cos(a)) - 5, int(c - (R + 15) * math.sin(a)) + 5)
        cv2.putText(img, lab, tp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    # V bar
    x0 = size + 12
    for i in range(2 * R):
        v = int(255 * (1 - i / (2 * R - 1)))
        img[pad + i, x0:x0 + bar - 14] = (v, v, v)
    cv2.rectangle(img, (x0, pad), (x0 + bar - 15, pad + 2 * R - 1), (90, 90, 90), 1)
    cv2.putText(img, "V", (x0 + 5, pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    _WHEEL_CACHE[key] = img
    return img.copy()


def wheel_xy(h, s, R=WHEEL_R, pad=26):
    """Pixel position of (OpenCV hue h, saturation s) on the wheel image."""
    a = math.radians(2.0 * h)
    r = R * s / 255.0
    c = R + pad
    return int(round(c + r * math.cos(a))), int(round(c - r * math.sin(a)))


def render_wheel(hsv=None, bgr=None, keep=None, mean_hsv=None, arc=None, s_rng=None, v_rng=None,
                 R=WHEEL_R, pad=26, bar=34, max_points=600):
    """Color wheel (CP-12). hsv, bgr: (N, 3) samples; keep: bool mask (False = excluded)."""
    img = _wheel_base(R, pad, bar)
    size = 2 * (R + pad)
    x0 = size + 12
    vy = lambda v: int(pad + (2 * R - 1) * (1 - v / 255.0))
    if hsv is not None and len(hsv):
        hsv = np.asarray(hsv)
        n = len(hsv)
        keep = np.ones(n, bool) if keep is None else np.asarray(keep, bool)
        idx = np.arange(n)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
        if arc is not None and s_rng is not None:
            lo, hi = arc
            span = (hi - lo) % 180
            steps = max(int(span * 2), 2)
            for s in s_rng:
                pts = [wheel_xy(lo + span * t / steps, s, R, pad) for t in range(steps + 1)]
                for col, t in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.polylines(img, [np.array(pts)], False, col, t, cv2.LINE_AA)
            for hh in (lo, hi):
                a, b = wheel_xy(hh, s_rng[0], R, pad), wheel_xy(hh, s_rng[1], R, pad)
                for col, t in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.line(img, a, b, col, t, cv2.LINE_AA)
        for i in idx:                          # excluded first, kept on top
            if keep[i]:
                continue
            p = wheel_xy(hsv[i, 0], hsv[i, 1], R, pad)
            cv2.circle(img, p, 4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(img, (x0 - 6, vy(hsv[i, 2])), (x0 - 1, vy(hsv[i, 2])), (0, 0, 255), 1)
        for i in idx:
            if not keep[i]:
                continue
            p = wheel_xy(hsv[i, 0], hsv[i, 1], R, pad)
            col = tuple(int(c) for c in bgr[i]) if bgr is not None else (255, 255, 255)
            cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, 4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.line(img, (x0 - 6, vy(hsv[i, 2])), (x0 - 1, vy(hsv[i, 2])), (230, 230, 230), 1)
        if v_rng is not None:
            xb = x0 + bar - 12
            for col, t in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.line(img, (xb, vy(v_rng[0])), (xb, vy(v_rng[1])), col, t)
                for v in v_rng:
                    cv2.line(img, (xb, vy(v)), (xb + 6, vy(v)), col, t)
        if mean_hsv is not None:
            p = wheel_xy(mean_hsv[0], mean_hsv[1], R, pad)
            for col, t in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, 16, t, cv2.LINE_AA)
            yv = vy(mean_hsv[2])
            cv2.drawMarker(img, (x0 + (bar - 14) // 2, yv), (0, 0, 255), cv2.MARKER_DIAMOND, 10, 2)
    return img


def encode(img, ext=".jpg", quality=80) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    return buf.tobytes() if ok else b""


def draw_measure(img, a, b, scale=1.0, color=(0, 255, 255)):
    """Points A and B of the focal-length measurement and the line between them (CP-20)."""
    pa = None if a is None else (int(round(a[0] * scale)), int(round(a[1] * scale)))
    pb = None if b is None else (int(round(b[0] * scale)), int(round(b[1] * scale)))
    for col, t in (((0, 0, 0), 3), (color, 1)):
        if pa and pb:
            cv2.line(img, pa, pb, col, t, cv2.LINE_AA)
        for p, lab in ((pa, "A"), (pb, "B")):
            if p:
                cv2.drawMarker(img, p, col, cv2.MARKER_TILTED_CROSS, 12, t, cv2.LINE_AA)
                cv2.putText(img, lab, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, t, cv2.LINE_AA)
    return img
