"""
Thin console_scripts shims.

Most kali_apps scripts define `main()` and parse `sys.argv` internally via
argparse, so they can be wired directly as console_scripts entry points.

`rf_distance.py` is the one exception: its `main(paths: list[str])` takes an
explicit list rather than reading sys.argv itself, so it needs a one-line
wrapper to work as a zero-argument entry point.
"""
import sys

from kali_apps.rf_distance import main as _rf_distance_main


def rf_distance_entrypoint() -> None:
    _rf_distance_main(sys.argv[1:])
