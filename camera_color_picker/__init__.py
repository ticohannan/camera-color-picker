"""Camera color picker for Jupyter: test USB cameras, measure colors and HSV ranges, estimate focal length.

    from camera_color_picker import ColorPicker
    picker = ColorPicker(save_dir="captures").show()

Design: docs/design.md (DD-CP). BSD-3-Clause. Author: ticohannan.
"""
__version__ = "0.2.0"

from .analysis import (cluster_stats, describe_pixel, disk_offsets, focal_from_dfov, focal_from_measurement,  # noqa: F401
                       fov_from_focal, hue_arc, hue_circular, inrange_pairs, orient, orient_point, unorient_point)
from .camera import Mode, SourceEntry, list_cameras, list_sources, parse_dshow_options  # noqa: F401


def __getattr__(name):
    if name == "ColorPicker":                       # ipywidgets is only needed for the widget
        from .widget import ColorPicker
        return ColorPicker
    raise AttributeError(name)
