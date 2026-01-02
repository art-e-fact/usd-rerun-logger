"""Gymnasium environment wrapper for logging training with Rerun.io."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, SupportsFloat

import rerun as rr

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

from .isaac_lab_logger import IsaacLabRerunLogger

__all__ = [
    "LogRerun",
]


class LogRerun(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType, RenderFrame],
    gym.utils.RecordConstructorArgs,
):
    """Logs Isaac Lab based environment episodes with Rerun.io.

    The API follows Gymnasium's :class:`gymnasium.wrappers.RecordVideo`
    wrapper, but instead of writing video, it logs Isaac Lab scenes to a Rerun stream.

    Typically you only want to log intermittently (e.g. every 100th episode or every
    N environment steps). Use either ``episode_trigger`` or ``step_trigger`` to decide
    when a new recording should start.

    The recording stream can either be provided directly via ``recording_stream``, or
    a new recording stream can be created that saves to ``save_path``. If neither is
    provided, it will try to find it using ``rr.get_data_recording()``. ``save_path``
    takes precedence over ``recording_stream``.

    Parameters
    ----------
    env:
        The environment to wrap. Must be Isaac Lab based and expose ``env.unwrapped.scene``.
    logged_envs:
        Indices of environments to log (for multi-env Isaac Lab scenes). If an ``int``,
        it is treated as a single environment index. If a list, those indices are logged.
        Defaults to ``0`` (the first environment).
    recording_stream:
        The Rerun recording stream to use. Ignored if ``save_path`` is provided.
    save_path:
        Path where the Rerun recording will be saved. If provided, a new recording stream
        is created that saves to this path.
    episode_trigger:
        Callable ``episode_trigger(episode_id) -> bool`` returning ``True`` iff a recording
        should start on the given episode. If both ``episode_trigger`` and ``step_trigger``
        are ``None``, defaults to
        :func:`gymnasium.utils.save_video.capped_cubic_video_schedule`.
        (Same signature as in :class:`gymnasium.wrappers.RecordVideo`.)
    step_trigger:
        Callable ``step_trigger(step_id) -> bool`` returning ``True`` iff a recording should
        start on the current *global* environment step (summed over episodes).
        (Same signature as in :class:`gymnasium.wrappers.RecordVideo`.)
    recording_length:
        Number of frames to record per snippet. If ``0``, record full episodes (until reset).
        If strictly positive, record fixed-length snippets.
        (Same as video_length in :class:`gymnasium.wrappers.RecordVideo`.)

    Raises
    ------
    ValueError
        If the environment does not expose a ``scene`` attribute on ``env.unwrapped``,
        i.e. it does not appear to be an Isaac Lab based environment.

    Notes
    -----
    - No vectorized wrapper variant is provided.
    - When a recording is active, each environment step logs the Isaac Lab scene at a
      timestamp derived from the scene ``physics_dt``.

    Examples
    --------
    Record every 10th episode and write `.rrd` files to disk:

    .. code-block:: python

       trigger = lambda ep: ep % 10 == 0

       rr.init("gymnasium_example", spawn=True)
       env = LogRerun(env, save_path="./save_rerun1.rrd", episode_trigger=trigger)

    Start a recording every 200th step and record 100 frames each time:

    .. code-block:: python
       trigger = lambda t: t % 200 == 0

       # LogRerun will find the global Rerun recording with rr.get_data_recording()
       rr.init("gymnasium_example", spawn=True)
       env = LogRerun(env, step_trigger=trigger, recording_length=100)

    Record everything, but split into chunks of 1000 frames:

    .. code-block:: python
       # Set the recording stream directly
       recording = rr.RecordingStream("gymnasium_example")
       env = LogRerun(env, recording_length=1000, recording_stream=recording)
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        logged_envs: int | list[int] = 0,
        recording_stream: rr.RecordingStream | None = None,
        save_path: Path | str | None = None,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        recording_length: int = 0,
    ):
        """Create the wrapper."""
        gym.utils.RecordConstructorArgs.__init__(
            self,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            recording_length=recording_length,
        )
        gym.Wrapper.__init__(self, env)

        # Check if the environment has a scene
        if not hasattr(self.env.unwrapped, "scene"):
            raise ValueError(
                (
                    "Cannot use LogRerun wrapper: the environment does not have a 'scene' attribute. "
                    "Are you sure this is an Isaac Lab based environment?"
                )
            )

        self.scene = self.env.unwrapped.scene

        self.logger = IsaacLabRerunLogger(
            scene=env.unwrapped.scene,
            logged_envs=logged_envs,
            recording_stream=recording_stream,
            save_path=save_path,
            application_id=env.spec.id if env.spec is not None else "env",
        )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.recording_length: int = (
            recording_length if recording_length != 0 else float("inf")
        )

        self.step_id = -1
        self.episode_id = -1
        self._timeline_name: str | None = None
        self._recorded_frames = 0

    def _capture_frame(self):
        """Capture a frame from the environment."""
        if self._timeline_name is None:
            return
        timestamp = self._recorded_frames * self.logger.scene.physics_dt
        self.logger.recording_stream.set_time(
            timeline=self._timeline_name, duration=timestamp
        )
        self.logger.log_scene()
        self._recorded_frames += 1

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"episode_{self.episode_id}")

        self._capture_frame()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action and logs the environment if recording is active."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"step_{self.step_id}")

        self._capture_frame()

        if self._recorded_frames > self.recording_length:
            self.stop_recording()

        return obs, rew, terminated, truncated, info

    def close(self):
        """Closes the wrapper and flushes the recording stream."""
        super().close()
        self.stop_recording()

    def start_recording(self, timeline_name: str):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        self._timeline_name = timeline_name
        self.logger.recording_stream.reset_time()
        self.logger.recording_stream.set_time(timeline=timeline_name, timestamp=0.0)

    def stop_recording(self):
        """Stop current recording and flush the recording stream."""
        self._timeline_name = None
        self._recorded_frames = 0
        self.logger.recording_stream.flush()
