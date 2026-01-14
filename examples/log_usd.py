"""Example: Log any USD file to Rerun.io.

Usage:
    python log_usd.py              # List available presets
    python log_usd.py orange       # Log a preset USD
    python log_usd.py path/to.usd  # Log a custom USD file
"""

import argparse
from pathlib import Path

from pxr import Usd
import rerun as rr

from usd_rerun_logger import UsdRerunLogger

# ============================================================================
# Core example logic
# ============================================================================


def log_usd_file(usd_path: str | Path) -> None:
    """Log a USD file to Rerun.

    This is the minimal code needed to visualize a USD stage:

        stage = Usd.Stage.Open(str(usd_path))
        logger = UsdRerunLogger(stage)
        logger.log_stage()
    """
    stage = Usd.Stage.Open(str(usd_path))
    logger = UsdRerunLogger(stage)
    logger.log_stage()


# ============================================================================
# CLI boilerplate
# ============================================================================

ASSETS_DIR = Path(__file__).parent / "assets"

# Preset USD files bundled with the examples
PRESETS: dict[str, Path] = {
    "billboard": ASSETS_DIR / "billboard.usda",
    "colored_cube": ASSETS_DIR / "colored_cube.usda",
    "orange": ASSETS_DIR / "Orange001" / "Orange001.usda",
    "so101": ASSETS_DIR / "so101_follower.usda",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log a USD file to Rerun.io for visualization.",
        epilog=f"Available presets: {', '.join(PRESETS.keys())}",
    )
    parser.add_argument(
        "usd",
        nargs="?",
        default=None,
        help="Preset name or path to a .usd/.usda/.usdc file.",
    )
    parser.add_argument(
        "--rrd",
        type=str,
        default=None,
        help="Path to save .rrd file instead of spawning viewer.",
    )
    return parser.parse_args()


def resolve_usd_path(usd_arg: str | None) -> Path:
    """Resolve preset name or file path to an actual Path."""
    if usd_arg is None:
        print("Available presets:")
        for name, path in PRESETS.items():
            status = "✓" if path.exists() else "✗ (missing)"
            print(f"  {name:15} {status}")
        print("\nUsage: python log_usd.py <preset_or_path>")
        raise SystemExit(0)

    # Check if it's a preset name
    if usd_arg in PRESETS:
        return PRESETS[usd_arg]

    # Otherwise treat as file path
    path = Path(usd_arg)
    if not path.exists():
        raise FileNotFoundError(f"USD file not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    usd_path = resolve_usd_path(args.usd)

    # Initialize Rerun viewer
    app_id = usd_path.stem + "_example"
    rr.init(app_id, spawn=args.rrd is None)

    # Log the USD (this is the example!)
    log_usd_file(usd_path)

    # Save to file if requested
    if args.rrd is not None:
        rr.save(args.rrd)


if __name__ == "__main__":
    main()
