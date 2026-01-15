"""Example: Log Isaac Lab environments to Rerun.io.

Usage:
    python isaac_lab_env.py              # List available presets
    python isaac_lab_env.py h1           # Log the H1 humanoid environment
    python isaac_lab_env.py go2          # Log the Unitree Go2 environment
"""

import argparse

import gymnasium as gym
import rerun as rr
import torch
from isaaclab.app import AppLauncher

# ============================================================================
# Environment presets (defined early for --help epilog)
# ============================================================================

PRESET_NAMES = [
    "franka-reach",
    "franka-cabinet",
    "peg-insert",
    "digit",
    "anymal-d",
    "spot",
    "a1",
    "go2",
    "h1",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Log Isaac Lab environments to Rerun.io for visualization.",
        epilog=f"Available presets: {', '.join(PRESET_NAMES)}",
    )
    parser.add_argument(
        "env",
        nargs="?",
        default=None,
        help="Preset name for the Isaac Lab environment.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps to run (default: 100).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no Isaac Sim GUI).",
    )
    parser.add_argument(
        "--rrd",
        type=str,
        default=None,
        help="Path to save .rrd file instead of spawning viewer.",
    )
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(headless=args.headless)

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import (  # noqa: E402
    UnitreeA1FlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (  # noqa: E402
    AnymalDFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.digit.flat_env_cfg import (  # noqa: E402
    DigitFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (  # noqa: E402
    UnitreeGo2FlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import (  # noqa: E402
    H1FlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (  # noqa: E402
    SpotFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import (  # noqa: E402
    FrankaCabinetEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (  # noqa: E402
    FrankaReachEnvCfg,
)

from usd_rerun_logger.env_wrapper import LogRerun  # noqa: E402

# ============================================================================
# Environment presets
# ============================================================================

PRESETS: dict[str, tuple[str, type]] = {
    "franka-reach": ("Isaac-Reach-Franka-v0", FrankaReachEnvCfg),
    "franka-cabinet": ("Isaac-Open-Drawer-Franka-Play-v0", FrankaCabinetEnvCfg),
    "peg-insert": ("Isaac-Factory-PegInsert-Direct-v0", FactoryTaskPegInsertCfg),
    "digit": ("Isaac-Velocity-Flat-Digit-Play-v0", DigitFlatEnvCfg_PLAY),
    "anymal-d": ("Isaac-Velocity-Flat-Anymal-D-Play-v0", AnymalDFlatEnvCfg_PLAY),
    "spot": ("Isaac-Velocity-Flat-Spot-Play-v0", SpotFlatEnvCfg_PLAY),
    "a1": ("Isaac-Velocity-Flat-Unitree-A1-Play-v0", UnitreeA1FlatEnvCfg_PLAY),
    "go2": ("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", UnitreeGo2FlatEnvCfg_PLAY),
    "h1": ("Isaac-Velocity-Rough-H1-Play-v0", H1FlatEnvCfg_PLAY),
}

# ============================================================================
# Core example logic
# ============================================================================


def run_env(env_id: str, cfg_class: type, steps: int = 100) -> None:
    """Run an Isaac Lab environment and log to Rerun.

    Args:
        env_id: The Gymnasium environment ID.
        cfg_class: The configuration class for the environment.
        steps: Number of simulation steps to run.
    """
    env = gym.make(env_id, cfg=cfg_class())
    env = LogRerun(env)
    env.reset()

    for _ in range(steps):
        action_np = env.action_space.sample()
        action = torch.as_tensor(action_np)
        env.step(action)


# ============================================================================
# CLI boilerplate
# ============================================================================


def resolve_env(env_arg: str | None) -> tuple[str, type]:
    """Resolve preset name to environment ID and config class."""
    if env_arg is None:
        print("Available environment presets:")
        for name, (env_id, _) in PRESETS.items():
            print(f"  {name:20} -> {env_id}")
        print("\nUsage: python isaac_lab_env.py <preset>")
        print("       python isaac_lab_env.py <preset> --steps 200")
        raise SystemExit(0)

    if env_arg not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{env_arg}'. Available: {available}")

    return PRESETS[env_arg]


def main() -> None:
    env_id, cfg_class = resolve_env(args.env)

    # Initialize Rerun viewer
    app_id = args.env + "_example"
    rr.init(app_id, spawn=args.rrd is None)

    # Run the environment
    run_env(env_id, cfg_class, steps=args.steps)

    # Save to file if requested
    if args.rrd is not None:
        rr.save(args.rrd)


if __name__ == "__main__":
    main()
