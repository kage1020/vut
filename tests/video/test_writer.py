import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from pytest_mock import MockerFixture

from vut.video.writer import VideoWriter


@pytest.fixture
def mock_ffmpeg(mocker: MockerFixture):
    """Mock ffmpeg to avoid dependency on ffmpeg installation"""
    mock_stdin = mocker.Mock()
    mock_stdin.write = mocker.Mock()
    mock_stdin.close = mocker.Mock()

    mock_process = mocker.Mock()
    mock_process.stdin = mock_stdin
    mock_process.wait = mocker.Mock()

    mock_stream = mocker.Mock()
    mock_stream.output = mocker.Mock(return_value=mock_stream)
    mock_stream.overwrite_output = mocker.Mock(return_value=mock_stream)
    mock_stream.run_async = mocker.Mock(return_value=mock_process)

    mocker.patch("ffmpeg.input", return_value=mock_stream)

    return {
        "stdin": mock_stdin,
        "process": mock_process,
        "stream": mock_stream,
    }


def test_video_writer__init():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name

    try:
        with VideoWriter(filename, framerate=30, size=(640, 480)) as writer:
            assert writer.filename == filename
            assert writer.framerate == 30
            assert writer.width == 640
            assert writer.height == 480
            assert writer.maxsize == 640
            assert writer.quality == 28
            assert writer.pix_fmt_in is None
            assert writer.pix_fmt_out == "yuv420p"
            assert writer.out is None

        assert os.path.exists(filename)
    finally:
        os.remove(filename)


def test_video_writer__resize():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name

    try:
        with VideoWriter(filename, framerate=30, size=(640, 480)) as writer:
            image = Image.new("RGB", (800, 600), color="blue")
            resized_image = writer._resize(image, max_size=640, image_size=image.size)

            assert resized_image.shape == (640, 480, 3)
            assert np.mean(resized_image) > 0
    finally:
        os.remove(filename)


def test_video_writer__update(mock_ffmpeg):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name

    try:
        with VideoWriter(filename, framerate=30, size=(640, 480)) as writer:
            image = Image.new("RGB", (640, 480), color="red")
            writer.update(image)

            assert mock_ffmpeg["stream"].output.called
            assert mock_ffmpeg["stream"].overwrite_output.called
            assert mock_ffmpeg["stream"].run_async.called
            assert mock_ffmpeg["stdin"].write.called

        assert os.path.exists(filename)
    finally:
        os.remove(filename)
