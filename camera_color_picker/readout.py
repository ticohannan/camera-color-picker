"""Text and table shown under the video (CP-8, CP-10, CP-20)."""
import math

from .analysis import corner_to_center, fov_from_focal


def pin_text(x, y, w, h, d) -> str:
    xc, yc = corner_to_center(x, y, w, h)
    return (f"mode=pin  frame={w}x{h}\n"
            f"xy_corner=({x}, {y})  xy_center=({xc}, {yc})   (x right, y down)\n"
            f"rgb={d['rgb']}  hex={d['hex']}\n"
            f"hsv_opencv={d['hsv_cv']}   (H 0-179, S 0-255, V 0-255)\n"
            f"hsv_deg_pct=({d['hsv_deg_pct'][0]} deg, {d['hsv_deg_pct'][1]}%, {d['hsv_deg_pct'][2]}%)\n"
            f"hsv_matlab={d['hsv_matlab']}   (all 0-1)")


def cluster_text(x, y, w, h, diameter, pts, st, exclude, k) -> str:
    if not st or st.get("n", 0) == 0:
        return "mode=cluster  no points inside the frame"
    xc, yc = corner_to_center(x, y, w, h)
    xs, ys = pts[:, 0], pts[:, 1]
    r = st["rows"]
    mb = st["mean_bgr"]
    lines = [f"mode=cluster  frame={w}x{h}  diameter={diameter}px  points kept {st['kept']} of {st['n']}"
             + (f" (excluded beyond {k:g} sd)" if exclude else ""),
             f"center xy_corner=({x}, {y})  xy_center=({xc}, {yc})   (x right, y down)",
             f"range  x {xs.min()}..{xs.max()}  y {ys.min()}..{ys.max()} (corner)   "
             f"x {xs.min() - w // 2}..{xs.max() - w // 2}  y {ys.min() - h // 2}..{ys.max() - h // 2} (center)",
             f"mean rgb=({mb[2]}, {mb[1]}, {mb[0]})  hex=#{mb[2]:02X}{mb[1]:02X}{mb[0]:02X}",
             f"mean hsv_opencv=({r['H']['mean']:.1f}, {r['S']['mean']:.1f}, {r['V']['mean']:.1f})  "
             f"sd=({r['H']['std']:.1f}, {r['S']['std']:.1f}, {r['V']['std']:.1f})",
             "# OpenCV inRange (HSV, min..max of kept points)" + ("; hue wraps: OR the two masks" if st["wraps"] else "")]
    for i, (lo, hi) in enumerate(st["inrange"]):
        sfx = "" if i == 0 else str(i + 1)
        lines.append(f"lower{sfx} = {lo}; upper{sfx} = {hi}")
    if st["low_sat"]:
        lines.append("# WARNING: mean S < 40 - nearly gray, hue is unreliable here")
    return "\n".join(lines)


def stats_html(st) -> str:
    if not st or st.get("n", 0) == 0:
        return ""
    head = "<tr><th></th><th>mean</th><th>sd</th><th>min</th><th>max</th></tr>"
    body = ""
    for c in "HSVRGB":
        r = st["rows"][c]
        note = " (circular; min..max is the arc)" if c == "H" else ""
        body += (f"<tr><td><b>{c}</b>{note}</td><td>{r['mean']:.1f}</td><td>{r['std']:.1f}</td>"
                 f"<td>{r['min']}</td><td>{r['max']}</td></tr>")
    return (f"<table style='font-family:monospace;border-collapse:collapse' cellpadding='3' border='1'>"
            f"{head}{body}</table><div style='font-size:85%'>H, S, V in OpenCV units; R, G, B 0-255.</div>")


def focal_text(w, h, a=None, b=None, length_mm=0.0, dist_mm=0.0, f_meas=None, dfov=0.0, f_sheet=None,
               pitch_um=0.0) -> str:
    """Lens readout (CP-20)."""
    lines = [f"frame {w}x{h}  (focal length in pixels is for this resolution)"]
    if a is not None and b is not None:
        p = math.hypot(b[0] - a[0], b[1] - a[1])
        lines.append(f"A=({a[0]}, {a[1]})  B=({b[0]}, {b[1]})  |AB| = {p:.1f} px")
    else:
        lines.append("A/B: put the crosshair on each end of an object of known length, press Set A, Set B")
    for name, f in (("measured (L, Z)", f_meas), ("datasheet diagonal FOV", f_sheet)):
        if f:
            fv = fov_from_focal(f, w, h)
            mm = f"  f = {f * pitch_um / 1000:.2f} mm (pitch {pitch_um:g} um)" if pitch_um > 0 else ""
            lines.append(f"{name}: f = {f:.1f} px{mm}")
            lines.append(f"   FOV h {fv['h']:.1f} deg, v {fv['v']:.1f} deg, d {fv['d']:.1f} deg;"
                         f" {fv['px_per_deg']:.2f} px/deg at center")
    if not f_meas and a is not None and b is not None:
        lines.append("enter L (mm) and Z (mm) to compute f")
    if f_meas:
        lines.append(f"copy: f_px={f_meas:.1f} @ {w}x{h}  (L={length_mm:g} mm, Z={dist_mm:g} mm)")
    return "\n".join(lines)


def props_html(props: dict) -> str:
    """Camera properties table (CP-19)."""
    if not props:
        return "<i>start a source to read its properties</i>"
    rows = "".join(f"<tr><td>{k}</td><td>{'not reported' if v is None else v}</td></tr>" for k, v in props.items())
    return ("<table style='font-family:monospace;border-collapse:collapse' cellpadding='2' border='1'>"
            f"{rows}</table><div style='font-size:85%'>Raw driver units (DirectShow: exposure is usually "
            "log2 seconds, focus a motor position). Webcams do not report the lens focal length; measure it below.</div>")
