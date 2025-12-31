"""Behavioral tests for the LogRerun wrapper."""

from __future__ import annotations

import sys
import types
from collections import defaultdict
from unittest import mock

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

# --- Lightweight Isaac Lab & logger stubs -------------------------------------------------


class TinyInteractiveScene:
    """Minimal stand-in for isaaclab.scene.InteractiveScene."""

    def __init__(self, physics_dt: float = 1 / 60.0):
        self.physics_dt = physics_dt
        self.num_envs = 1


def _install_isaaclab_stub() -> None:
    if "isaaclab.scene" in sys.modules:
        return

    isaaclab_mod = types.ModuleType("isaaclab")
    scene_mod = types.ModuleType("isaaclab.scene")
    scene_mod.InteractiveScene = TinyInteractiveScene
    isaaclab_mod.scene = scene_mod

    sys.modules["isaaclab"] = isaaclab_mod
    sys.modules["isaaclab.scene"] = scene_mod


_install_isaaclab_stub()
from  usd_rerun_logger.env_logger import LogRerun  # noqa: E402

# --- Minimal Isaac-like environment -------------------------------------------------------


class DummyIsaacEnv(gym.Env):
    """Simple environment with a scene attribute for LogRerun."""

    def __init__(self, max_steps: int = 4):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._max_steps = max_steps
        self._step_count = 0
        self.scene = TinyInteractiveScene()
        self._last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._last_obs.fill(0.0)
        return self._last_obs.copy(), {}

    def step(self, action):
        self._step_count += 1
        self._last_obs[:] = self._step_count
        terminated = self._step_count >= self._max_steps
        if terminated:
            self._step_count = 0
        return self._last_obs.copy(), 1.0, terminated, False, {}

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        return None


# --- Tests -------------------------------------------------------------------------------


@mock.patch("usd_rerun_logger.env_logger.IsaacLabRerunLogger", autospec=True)
@mock.patch("rerun.RecordingStream", autospec=True)
def test_episode_trigger_logs_only_selected_episodes(_, mock_recording_stream):
    log_episodes = {0, 2}
    wrapper = LogRerun(
        DummyIsaacEnv(max_steps=3),
        episode_trigger=lambda episode: episode in log_episodes,
        step_trigger=lambda _: False,
        recording_stream=mock_recording_stream,
    )
    wrapper.logger.scene = wrapper.scene
    recording = wrapper.logger.recording_stream

    for episode in range(4):
        wrapper.reset()
        log_episode = episode in log_episodes
        assert recording.reset_time.called == log_episode

        for step in range(3):
            _, _, terminated, _, _ = wrapper.step(0)

            if log_episode:
                assert recording.set_time.call_args.kwargs == {
                    "timeline": f"episode_{episode}",
                    "duration": pytest.approx((step + 1) * wrapper.scene.physics_dt),
                }
            else:
                recording.set_time.assert_not_called()

            if terminated:
                break

        recording.reset_mock()

    wrapper.close()


# def test_step_trigger_emits_timelines_on_schedule():
#     wrapper = LogRerun(
#         DummyIsaacEnv(max_steps=5),
#         episode_trigger=lambda _: False,
#         step_trigger=lambda step: step in (0, 3),
#     )

#     wrapper.reset()
#     for _ in range(8):
#         wrapper.step(0)

#     frames = wrapper.logger.frames_by_timeline
#     assert {"step_0", "step_3"}.issubset(frames)
#     assert frames["step_0"] > 0
#     assert frames["step_3"] > 0

#     wrapper.close()


# def test_recording_length_stops_after_cap():
#     recording_length = 1
#     wrapper = LogRerun(
#         DummyIsaacEnv(max_steps=5),
#         episode_trigger=lambda _: False,
#         step_trigger=lambda step: step == 0,
#         recording_length=recording_length,
#     )

#     wrapper.reset()
#     for _ in range(4):
#         wrapper.step(0)

#     frames = wrapper.logger.frames_by_timeline
#     assert frames["step_0"] == recording_length
#     assert wrapper.logger.recording_stream.flush_calls == 1
#     assert wrapper.timeline_name is None
#     assert wrapper.recorded_frames == 0

#     wrapper.close()
