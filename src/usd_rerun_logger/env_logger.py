"""Gymnasium environment wrapper for logging training with Rerun.io."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, SupportsFloat

import numpy as np
import rerun as rr

import gymnasium as gym
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled, InvalidProbability

from .util import get_recording_stream
from .isaac_lab_logger import IsaacLabRerunLogger

from gymnasium.wrappers.rendering import RecordVideo
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
    This function starts a video at every episode that is a power of 3 until 1000 and then every 1000 episodes.
    By default, the logging will be stopped once reset is called.
    However, you can also create recordings of fixed length (possibly spanning several episodes)
    by passing a strictly positive value for ``recording_length``.

    No vector version of the wrapper exists.

    Examples - Run the environment for 50 episodes, and save the video every 10 episodes starting from the 0th:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 10 == 0
        >>> env = RecordVideo(env, video_folder="./save_videos1", episode_trigger=trigger, disable_logger=True)
        >>> for i in range(50):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos1"))
        5

    Examples - Run the environment for 5 episodes, start a recording every 200th step, making sure each video is 100 frames long:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 200 == 0
        >>> env = RecordVideo(env, video_folder="./save_videos2", step_trigger=trigger, recording_length=100, disable_logger=True)
        >>> for i in range(5):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     _ = env.action_space.seed(123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos2"))
        2

    Examples - Run 3 episodes, record everything, but in chunks of 1000 frames:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> env = RecordVideo(env, video_folder="./save_videos3", recording_length=1000, disable_logger=True)
        >>> for i in range(3):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos3"))
        2

    Change logs:
     * v0.25.0 - Initially added to replace ``wrappers.monitoring.VideoRecorder``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        recording_stream: rr.RecordingStream | None = None,
        save_path: Path | str | None = None,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        recording_length: int = 0,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            recording_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            fps (int): The frame per second in the video. Provides a custom video fps for environment, if ``None`` then
                the environment metadata ``render_fps`` key is used if it exists, otherwise a default value of 30 is used.
            gc_trigger: Function that accepts an integer and returns ``True`` iff garbage collection should be performed after this episode
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            recording_length=recording_length,
        )
        gym.Wrapper.__init__(self, env)


        self.logger = IsaacLabRerunLogger(
            scene=env.unwrapped.scene,
            recording_stream=recording_stream,
            save_path=save_path,
            application_id=env.spec.id if env.spec is not None else "env",
        )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.gc_trigger = gc_trigger
        self.recording_length: int = recording_length if recording_length != 0 else float("inf")

        self.step_id = -1
        self.episode_id = -1

    def _capture_frame(self):
        """Capture a frame from the environment."""
        self.logger.log_scene()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"episode-{self.episode_id}")

        self._capture_frame()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"step-{self.step_id}")

        self._capture_frame()


        return obs, rew, terminated, truncated, info

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        self.logger.recording_stream.reset_time()

    def stop_recording(self):
        """Stop current recording and saves the video."""
        self.logger.recording_stream.flush()
