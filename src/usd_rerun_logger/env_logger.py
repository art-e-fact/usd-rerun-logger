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

    Based on Gymnasium's :class:`RecordVideo` wrapper: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo

    Usually, you only want to log episodes intermittently, say every hundredth episode or at every thousandth environment step.
    To do this, you can specify ``episode_trigger`` or ``step_trigger``.
    They should be functions returning a boolean that indicates whether a logging should be started at the
    current episode or step, respectively.

    The ``episode_trigger`` should return ``True`` on the episode when logging should start.
    The ``step_trigger`` should return ``True`` on the n-th environment step that the logging should be started, where n sums over all previous episodes.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed, i.e. :func:`capped_cubic_video_schedule`.
    This function starts a recording at every episode that is a power of 3 until 1000 and then every 1000 episodes.
    By default, the logging will be stopped once reset is called.
    However, you can also create recordings of fixed length (possibly spanning several episodes)
    by passing a strictly positive value for ``recording_length``.

    No vector version of the wrapper exists.

    Examples - Run the environment for 50 episodes, and save the recording every 10 episodes starting from the 0th:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 10 == 0
        >>> env = LogRerun(env, save_path="./save_rerun1", episode_trigger=trigger)
        >>> for i in range(50):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_rerun1"))
        5

    Examples - Run the environment for 5 episodes, start a recording every 200th step, making sure each recording is 100 frames long:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 200 == 0
        >>> env = LogRerun(env, save_path="./save_rerun2", step_trigger=trigger, recording_length=100)
        >>> for i in range(5):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     _ = env.action_space.seed(123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_rerun2"))
        2

    Examples - Run 3 episodes, record everything, but in chunks of 1000 frames:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> env = LogRerun(env, save_path="./save_rerun3", recording_length=1000)
        >>> for i in range(3):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_rerun3"))
        2
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
        """Wrapper records Rerun logs of rollouts.

        Args:
            env: The environment that will be wrapped
            logged_envs: The indices of the environments to log.
            recording_stream: The Rerun recording stream to use. If ``None``, a new stream will be created.
            save_path: The path where the Rerun recording will be saved. If ``None``, the recording will not be saved to disk.
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            recording_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            recording_length=recording_length,
        )
        gym.Wrapper.__init__(self, env)

        # Check if the environment has a scene
        if not hasattr(self.env.unwrapped, "scene"):
            raise ValueError((
                "Cannot use LogRerun wrapper: the environment does not have a 'scene' attribute. "
                "Are you sure this is an Isaac Lab based environment?"
            ))

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
        print(f"Starting Rerun recording: {timeline_name}")
        self.logger.recording_stream.set_time(timeline_name, timestamp=0.0)

    def stop_recording(self):
        """Stop current recording and flush the recording stream."""
        self._timeline_name = None
        self._recorded_frames = 0
        self.logger.recording_stream.flush()
