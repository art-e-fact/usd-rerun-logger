"""Utility helpers for usd-rerun-logger."""

import hashlib
import importlib
import os
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
from PIL import Image
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


def _get_cache_dir() -> str:
    """Get or create the usd_rerun_logger cache directory in the system's tmp folder."""
    cache_dir = os.path.join(tempfile.gettempdir(), "usd_rerun_logger")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_image(url: str) -> str | None:
    """
    Download an image from a URL to the cache directory.
    Returns the local file path if successful, None otherwise.
    """
    cache_dir = _get_cache_dir()

    # Create a unique filename based on the URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # Try to preserve the original extension
    parsed = urlparse(url)
    path_parts = parsed.path.rsplit(".", 1)
    ext = f".{path_parts[1]}" if len(path_parts) > 1 else ""
    cached_path = os.path.join(cache_dir, f"{url_hash}{ext}")

    # Return cached file if it already exists
    if os.path.exists(cached_path):
        return cached_path

    try:
        # Download the file
        with urlopen(url, timeout=30) as response:
            data = response.read()

        # Write to a temporary file first, then verify it's an image
        temp_path = cached_path + ".tmp"
        with open(temp_path, "wb") as f:
            f.write(data)

        # Verify it's a valid image using PIL
        try:
            with Image.open(temp_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            print(f"Downloaded file is not a valid image: {url} - {e}")
            os.remove(temp_path)
            return None

        # Move temp file to final location
        os.rename(temp_path, cached_path)
        return cached_path

    except URLError as e:
        print(f"Failed to download image from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False
