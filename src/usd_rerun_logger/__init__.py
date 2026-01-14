"""Rerun.io logger for USD and NVIDIA Omniverse apps."""

from importlib.metadata import PackageNotFoundError, version

from .env_wrapper import LogRerun
from .isaac_lab_logger import IsaacLabRerunLogger
from .usd_logger import UsdRerunLogger

try:
    __version__ = version("usd-rerun-logger")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "IsaacLabRerunLogger",
    "LogRerun",
    "UsdRerunLogger",
]
