"""Coordinates, orientation, color statistics, sampling and focal length (DD-CP sections 5, 7, 9)."""
import math
from typing import Tuple

import cv2
import numpy as np


def corner_to_center(x, y, w, h):
    return x - w // 2, y - h // 2


def center_to_corner(xc, yc, w, h):
    return xc + w // 2, yc + h // 2


def clamp_xy(x, y, w, h):
    return int(min(max(round(x), 0), w - 1)), int(min(max(round(y), 0), h - 1))


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 BGR -> (N, 3) uint8 HSV, OpenCV units (H 0-179, S and V 0-255)."""
    a = np.asarray(bgr, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_BGR2HSV).reshape(-1, 3)


def hue_circular(h: np.ndarray) -> Tuple[float, float, float]:
    """Circular mean and std of OpenCV hues (angle = 2h deg). Returns (mean 0-180, std in hue units, R)."""
    a = np.radians(np.asarray(h, float) * 2.0)
    C, S = float(np.mean(np.cos(a))), float(np.mean(np.sin(a)))
    R = math.hypot(C, S)
    mean_deg = math.degrees(math.atan2(S, C)) % 360.0     # class Atan2(x=C, y=S); Python atan2(y, x)
    std_deg = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))))
    return mean_deg / 2.0, std_deg / 2.0, R


def hue_diff(h, ref):
    """Signed hue difference h - ref in OpenCV units, wrapped into -90 ... <90."""
    return (np.asarray(h, float) - ref + 90.0) % 180.0 - 90.0


def hue_arc(h: np.ndarray) -> Tuple[int, int]:
    """Smallest arc containing all hues: (lo, hi). lo > hi means the arc wraps through 0/180."""
    u = np.unique(np.asarray(h, int) % 180)
    if len(u) == 1:
        return int(u[0]), int(u[0])
    gaps = np.diff(np.concatenate([u, [u[0] + 180]]))
    i = int(np.argmax(gaps))                       # largest empty gap is between u[i] and u[i+1]
    lo = int(u[(i + 1) % len(u)])
    hi = int(u[i])
    return lo, hi


def inrange_pairs(h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    """OpenCV inRange (lower, upper) pairs; two when the hue arc wraps."""
    if h_lo <= h_hi:
        return [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]
    return [((h_lo, s_lo, v_lo), (179, s_hi, v_hi)), ((0, s_lo, v_lo), (h_hi, s_hi, v_hi))]


def describe_pixel(bgr) -> dict:
    b, g, r = (int(v) for v in bgr)
    h, s, v = (int(x) for x in bgr_to_hsv(np.array([[b, g, r]]))[0])
    return {"rgb": (r, g, b), "hex": f"#{r:02X}{g:02X}{b:02X}", "hsv_cv": (h, s, v),
            "hsv_deg_pct": (round(h * 2.0, 1), round(s / 2.55, 1), round(v / 2.55, 1)),
            "hsv_matlab": (round(h / 180.0, 3), round(s / 255.0, 3), round(v / 255.0, 3))}


def cluster_stats(bgr_samples: np.ndarray, exclude: bool = False, k: float = 2.0) -> dict:
    """Statistics of sampled BGR pixels (DD-CP section 5)."""
    bgr = np.asarray(bgr_samples, np.uint8).reshape(-1, 3)
    n = len(bgr)
    if n == 0:
        return {"n": 0}
    hsv = bgr_to_hsv(bgr).astype(float)
    keep = np.ones(n, bool)
    if exclude and n > 2:
        hm, hs, _ = hue_circular(hsv[:, 0])
        z = np.stack([np.abs(hue_diff(hsv[:, 0], hm)) / max(hs, 1.0),
                      np.abs(hsv[:, 1] - hsv[:, 1].mean()) / max(hsv[:, 1].std(), 1.0),
                      np.abs(hsv[:, 2] - hsv[:, 2].mean()) / max(hsv[:, 2].std(), 1.0)], 1)
        keep = (z <= k).all(1)
        if not keep.any():
            keep[:] = True
    kh, kb = hsv[keep], bgr[keep].astype(float)
    hm, hs, R = hue_circular(kh[:, 0])
    h_lo, h_hi = hue_arc(kh[:, 0])
    rows = {"H": {"mean": hm, "std": hs, "min": h_lo, "max": h_hi}}
    for i, c in ((1, "S"), (2, "V")):
        rows[c] = {"mean": kh[:, i].mean(), "std": kh[:, i].std(), "min": int(kh[:, i].min()), "max": int(kh[:, i].max())}
    for i, c in ((2, "R"), (1, "G"), (0, "B")):
        rows[c] = {"mean": kb[:, i].mean(), "std": kb[:, i].std(), "min": int(kb[:, i].min()), "max": int(kb[:, i].max())}
    mean_bgr = tuple(int(round(v)) for v in kb.mean(0))
    pairs = inrange_pairs(h_lo, h_hi, rows["S"]["min"], rows["S"]["max"], rows["V"]["min"], rows["V"]["max"])
    return {"n": n, "kept": int(keep.sum()), "keep": keep, "hsv": hsv.astype(int), "rows": rows,
            "mean_bgr": mean_bgr, "mean_hsv": (hm, rows["S"]["mean"], rows["V"]["mean"]), "R": R,
            "wraps": h_lo > h_hi, "inrange": pairs, "low_sat": rows["S"]["mean"] < 40}


def disk_offsets(n: int, diameter: float, pattern: str = "random", seed: int = 0) -> np.ndarray:
    """Integer (dx, dy) offsets inside a disc of the given diameter; duplicates removed; (0, 0) always first."""
    R = max(diameter, 1.0) / 2.0
    n = max(int(n), 1)
    if pattern == "even":
        k = np.arange(n)
        r = R * np.sqrt((k + 0.5) / n)
        phi = np.radians(k * 137.508)
    else:
        rng = np.random.default_rng(seed)
        r = R * np.sqrt(rng.random(n))
        phi = 2 * np.pi * rng.random(n)
    off = np.stack([np.rint(r * np.cos(phi)), np.rint(r * np.sin(phi))], 1).astype(int)
    off = off[(off ** 2).sum(1) <= R * R + 1e-9]
    off = np.vstack([[0, 0], off])
    _, idx = np.unique(off, axis=0, return_index=True)
    return off[np.sort(idx)]


def sample_points(w, h, x, y, offsets) -> np.ndarray:
    p = offsets + np.array([x, y])
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    return p[ok]


# =============================================================== orientation (CP-18, DD-CP section 7)
ROTATIONS = (0, 90, 180, 270)
_CV_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def orient(img, rot=0, flip_h=False, flip_v=False):
    """Turn a camera frame: flips first, then a clockwise rotation of 0, 90, 180 or 270 degrees."""
    if flip_h and flip_v:
        img = cv2.flip(img, -1)
    elif flip_h:
        img = cv2.flip(img, 1)
    elif flip_v:
        img = cv2.flip(img, 0)
    rot = int(rot) % 360
    if rot:
        img = cv2.rotate(img, _CV_ROT[rot])
    return img


def oriented_size(w, h, rot=0):
    return (h, w) if int(rot) % 180 == 90 else (w, h)


def orient_point(x, y, w, h, rot=0, flip_h=False, flip_v=False):
    """Camera pixel (x, y) in a w x h frame -> pixel in the turned frame (same steps as orient)."""
    if flip_h:
        x = w - 1 - x
    if flip_v:
        y = h - 1 - y
    rot = int(rot) % 360
    if rot == 90:
        return h - 1 - y, x
    if rot == 180:
        return w - 1 - x, h - 1 - y
    if rot == 270:
        return y, w - 1 - x
    return x, y


def unorient_point(x, y, w, h, rot=0, flip_h=False, flip_v=False):
    """Inverse of orient_point. w, h are the CAMERA frame's size (before turning)."""
    rot = int(rot) % 360
    if rot == 90:
        x, y = y, h - 1 - x
    elif rot == 180:
        x, y = w - 1 - x, h - 1 - y
    elif rot == 270:
        x, y = w - 1 - y, x
    if flip_v:
        y = h - 1 - y
    if flip_h:
        x = w - 1 - x
    return x, y


# =============================================================== focal length (CP-20, DD-CP section 9)
def focal_from_measurement(p_px, length_mm, dist_mm):
    """Pinhole camera: an object of length L at distance Z spans p pixels -> f = p * Z / L (pixels)."""
    if p_px <= 0 or length_mm <= 0 or dist_mm <= 0:
        return None
    return p_px * dist_mm / length_mm


def focal_from_dfov(dfov_deg, w, h):
    """Diagonal field of view (deg) -> f in pixels at w x h."""
    if not 0 < dfov_deg < 180:
        return None
    return (math.hypot(w, h) / 2.0) / math.tan(math.radians(dfov_deg) / 2.0)


def fov_from_focal(f, w, h) -> dict:
    """Fields of view (deg) for focal length f (pixels) at w x h. FOV = 2 Atan2(x = f, y = half-size)."""
    fov = lambda half: math.degrees(2.0 * math.atan2(half, f))      # Python atan2(y, x)
    return {"h": fov(w / 2.0), "v": fov(h / 2.0), "d": fov(math.hypot(w, h) / 2.0),
            "px_per_deg": f * math.pi / 180.0}
