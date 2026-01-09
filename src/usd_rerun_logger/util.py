"""Utility helpers for usd-rerun-logger."""

import importlib
from pathlib import Path
import rerun as rr


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


def get_recording_stream(
    recording_stream: rr.RecordingStream | None = None,
    save_path: Path | str | None = None,
    application_id: str | None = None,
) -> rr.RecordingStream:
    """Tries to get or create the appropriate Rerun recording stream.

    If save_path is provided, a new recording stream is created and saved to that path.

    :param recording_stream: An optional existing recording stream to use.
    :param save_path: An optional path to save the recording stream.
    :param application_id: The application ID to use when creating a new recording stream.
    """
    recording_stream = rr.get_data_recording(recording_stream)
    if save_path is not None:
        if application_id is None:
            application_id = "usd_rerun_logger"
        recording_stream = rr.RecordingStream(application_id=application_id)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        recording_stream.save(path=save_path)

    if recording_stream is None:
        # recording stream or save path must be provided if the global one is not set
        raise ValueError(
            "No Rerun recording stream is set. Please provide either a recording stream, "
            "a save path, or start a global recording stream (e.g., via `rerun.init()`)."
        )

    return recording_stream
