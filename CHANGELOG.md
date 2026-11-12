# Changelog

| Date | Version | Change | Status |
| --- | --- | --- | --- |
| 2026-11-11 | 0.2.0 | Own repository, split into modules (`camera`, `analysis`, `drawing`, `readout`, `widget`). Video width slider and fit-to-cell; stacked or side-by-side layout; collapsible control groups; wheel size (CP-17). Flip and rotate (CP-18). Camera properties (CP-19). Focal length in pixels and field of view from two crosshair points or a datasheet FOV (CP-20). North American spelling. Design DD-CP v0.2 | 24 tests pass on Linux; checked in JupyterLab in a headless browser at 1366×768. Not yet run on Windows or with a real camera |
| 2026-10-22 | 0.1.0 | First version inside `rot3u-robots` (`host/rot3u_host/color_picker.py`): camera listing and modes, draggable crosshair, pin-point and cluster sampling, circular hue statistics, color wheel, captures | Tested with the test pattern only |
