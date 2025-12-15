"""Utility helpers for usd-rerun-logger."""

import importlib


def assert_usd_core_dependency() -> None:
    """Ensure that the pxr USD bindings are importable."""

    try:
        importlib.import_module("pxr")
    except ImportError as exc:  # pragma: no cover - depends on external install
        message = (
            "Unable to import `pxr`. If you are using Isaac Sim or Isaac Lab, "
            "call this check only after the Omniverse application is fully "
            "initialized. Otherwise install `usd-core` manually. We do not "
            "declare `usd-core` as a dependency because it conflicts with the "
            "pxr binaries bundled with Omniverse."
        )
        raise ImportError(message) from exc

def assert_isaac_lab_dependency() -> None:
    """Ensure that the isaaclab package is importable."""

    try:
        importlib.import_module("isaaclab")
    except ImportError as exc:  # pragma: no cover - depends on external install
        message = (
            "Unable to import `isaaclab`. Please ensure that you have Isaac Lab "
            "installed and that your PYTHONPATH is set up correctly."
        )
        raise ImportError(message) from exc