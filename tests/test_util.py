"""Tests for usd_rerun_logger.util module."""

from __future__ import annotations

from unittest import mock

import pytest

from usd_rerun_logger.util import get_recording_stream


def test_get_recording_stream_with_existing_stream():
    """Should return the provided recording stream."""
    stream = mock.Mock()
    with mock.patch("usd_rerun_logger.util.rr.get_data_recording") as get_data:
        get_data.return_value = stream

        result = get_recording_stream(recording_stream=stream)

        assert result is stream
        get_data.assert_called_once_with(stream)


@mock.patch("usd_rerun_logger.util.rr.RecordingStream")
@mock.patch("usd_rerun_logger.util.rr.get_data_recording")
def test_get_recording_stream_with_save_path(get_data, stream_class):
    """Should create and save a new stream when save_path is provided."""
    mock_stream = mock.Mock()
    get_data.return_value = None
    stream_class.return_value = mock_stream

    result = get_recording_stream(save_path="/tmp/recording.rrd")

    assert result is mock_stream
    stream_class.assert_called_once_with(application_id="usd_rerun_logger")
    mock_stream.save.assert_called_once()


@mock.patch("usd_rerun_logger.util.rr.RecordingStream")
@mock.patch("usd_rerun_logger.util.rr.get_data_recording")
def test_get_recording_stream_with_custom_application_id(get_data, stream_class):
    """Should use provided application_id when creating a stream."""
    mock_stream = mock.Mock()
    get_data.return_value = None
    stream_class.return_value = mock_stream

    result = get_recording_stream(
        save_path="/tmp/recording.rrd",
        application_id="my_app",
    )

    assert result is mock_stream
    stream_class.assert_called_once_with(application_id="my_app")


@mock.patch("usd_rerun_logger.util.rr.get_data_recording")
def test_get_recording_stream_returns_global_stream(get_data):
    """Should return global stream when no save_path is provided."""
    global_stream = mock.Mock()
    get_data.return_value = global_stream

    result = get_recording_stream()

    assert result is global_stream


@mock.patch("usd_rerun_logger.util.rr.get_data_recording")
def test_get_recording_stream_raises_without_stream(get_data):
    """Should raise ValueError when no recording stream is available."""
    get_data.return_value = None

    with pytest.raises(ValueError, match="No Rerun recording stream is set"):
        get_recording_stream()
