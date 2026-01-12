"""Behavioral tests for the LogRerun wrapper."""

from __future__ import annotations

import sys
import types
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import rerun as rr
from gymnasium import spaces

NUM_TIMELINES = 2  # Number of timelines created for each recording

# --- Lightweight Isaac Lab & logger stubs -------------------------------------------------


class TinyInteractiveScene:
    """Minimal stand-in for isaaclab.scene.InteractiveScene."""

    def __init__(self, physics_dt: float = 1 / 60.0):
        self.physics_dt = physics_dt
        self.num_envs = 1
        self.stage = None


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
from usd_rerun_logger.env_wrapper import LogRerun  # noqa: E402

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


@pytest.fixture()
def recording() -> rr.RecordingStream:  # pyright: ignore[reportInvalidTypeForm]
    with mock.patch(
        "usd_rerun_logger.util.rr.get_data_recording"
    ) as get_data_recording:
        recording = mock.Mock(spec=rr.RecordingStream)
        get_data_recording.return_value = recording
        yield recording


def test_episode_trigger_logs_only_selected_episodes(recording):
    """LogRerun should only log episodes selected by the episode trigger."""
    log_episodes = {0, 2}

    wrapper = LogRerun(
        DummyIsaacEnv(max_steps=4),
        episode_trigger=lambda episode: episode in log_episodes,
        step_trigger=lambda _: False,
    )

    for episode in range(4):
        wrapper.reset()
        log_episode = episode in log_episodes
        assert recording.reset_time.called == log_episode

        for step in range(6):
            if log_episode:
                assert recording.set_time.call_args_list[-2].kwargs == {
                    "timeline": f"episode_{episode}_timestamp",
                    "duration": pytest.approx(step * wrapper.scene.physics_dt),
                }
                assert recording.set_time.call_args_list[-1].kwargs == {
                    "timeline": f"episode_{episode}_step",
                    "sequence": step,
                }
            else:
                recording.set_time.assert_not_called()

            _, _, terminated, _, _ = wrapper.step(0)

            if terminated:
                break

        recording.reset_mock()

    wrapper.close()


# --- Migrated tests from Gymnasium ---------------------------------------------------------------
# These tests are adapted to make sure we're matching the RecordVideo trigger API.
# See the original tests at https://github.com/Farama-Foundation/Gymnasium/blob/ffb4c9f33144a79398e2c140207c98863b970ee7/tests/wrappers/test_record_video.py


@pytest.mark.parametrize("episodic_trigger", [None, lambda x: x in [0, 3, 5, 10, 12]])
def test_episodic_trigger(episodic_trigger, recording):
    """Test LogRerun using the default episode trigger."""
    env = DummyIsaacEnv(max_steps=30)
    env = LogRerun(env, episode_trigger=episodic_trigger)

    env.reset()
    episode_count = 0
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            episode_count += 1
    env.close()

    timelines = set(
        map(lambda call: call.kwargs["timeline"], recording.set_time.call_args_list)
    )
    assert env.episode_trigger is not None
    assert (
        len(timelines)
        == sum(env.episode_trigger(i) for i in range(episode_count + 1)) * NUM_TIMELINES
    )


def test_step_trigger(recording):
    """Test LogRerun defining step trigger function."""
    env = DummyIsaacEnv(max_steps=30)
    env = LogRerun(env, step_trigger=lambda x: x % 100 == 0)
    env.reset()
    total_steps = 0
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        total_steps += 1
        if terminated or truncated:
            env.reset()
    env.close()
    timelines = set(
        map(lambda call: call.kwargs["timeline"], recording.set_time.call_args_list)
    )
    assert len(timelines) == 2 * NUM_TIMELINES


def test_both_episodic_and_step_trigger(recording):
    """Test LogRerun defining both step and episode trigger functions."""
    env = DummyIsaacEnv(max_steps=30)
    env = LogRerun(
        env,
        step_trigger=lambda x: x == 100,
        episode_trigger=lambda x: x == 0 or x == 3,
    )
    env.reset(seed=123)
    env.action_space.seed(123)
    total_steps = 0
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        total_steps += 1
        if terminated or truncated:
            env.reset()
    env.close()
    timelines = set(
        map(lambda call: call.kwargs["timeline"], recording.set_time.call_args_list)
    )

    assert len(timelines) == 3 * NUM_TIMELINES


def test_video_length(recording, recording_length: int = 10):
    """Test if argument recording_length of LogRerun works properly."""
    env = DummyIsaacEnv(max_steps=20)
    env = LogRerun(
        env, step_trigger=lambda x: x == 0, recording_length=recording_length
    )

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(recording_length):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break

    # check that the environment is still recording then take a step to take the number of steps > recording length
    assert env._timeline_name is not None
    env.step(env.action_space.sample())
    assert env._timeline_name is None
    env.close()

    # check that only one recording is recorded
    timelines = set(
        map(lambda call: call.kwargs["timeline"], recording.set_time.call_args_list)
    )
    assert len(timelines) == 1 * NUM_TIMELINES
